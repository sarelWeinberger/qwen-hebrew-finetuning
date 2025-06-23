#!/usr/bin/env python3
"""
Gepeta EC2 Orchestrator - מנהל 143 מכונות EC2
"""

import boto3
import pandas as pd
import time
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# הגדרות EC2
INSTANCE_TYPE = "c6i.large"
REGION = "us-east-1"
AMI_ID = "ami-0c02fb55956c7d316"  # Ubuntu 20.04
KEY_PAIR_NAME = "gepeta-worker-key"
SECURITY_GROUP_NAME = "gepeta-workers"

# הגדרות S3
STATUS_BUCKET = "gepeta-datasets"
STATUS_PREFIX = "worker-status/"


class GepetaOrchestrator:
    def __init__(self):
        self.ec2_client = boto3.client('ec2', region_name=REGION)
        self.s3_client = boto3.client('s3')
        self.instances = []
        self.tasks = []

    def load_datasets_data(self):
        """טעינת נתוני datasets קבועים מהאקסל שסופק"""

        # נתונים מהאקסל - רק השורות עם Prefix
        DATASETS = [
            {'name': 'geektime', 'prefix': 'Geektime', 'num_files': 3},
            {'name': 'israel hayom', 'prefix': 'Yisrael', 'num_files': 12},
            {'name': 'booksnli2', 'prefix': 'Books', 'num_files': 28},
            {'name': 'tzenzura', 'prefix': 'AllTzen', 'num_files': 51},
            {'name': 'tweets', 'prefix': 'hebrew_tweets', 'num_files': 48},
            {'name': 'oscar', 'prefix': 'AllOfH', 'num_files': 1}
        ]

        print("📊 טוען נתוני datasets:")

        tasks = []
        for dataset in DATASETS:
            dataset_name = dataset['name'].lower().replace(' ', '')  # geektime, israelhayom, etc.
            prefix = dataset['prefix']
            num_files = dataset['num_files']

            print(f"  • {dataset['name'].title()} ({prefix}): {num_files} קבצים")

            # יצירת משימה לכל part
            for part_num in range(num_files):
                task = {
                    'dataset_name': dataset_name,
                    'prefix': prefix,
                    'part_number': part_num,
                    'task_id': f"{prefix}_part-{part_num}"
                }
                tasks.append(task)

        print(f"🎯 סה\"כ {len(tasks)} משימות")
        self.tasks = tasks
        return tasks

    def create_user_data_script(self, task):
        """יצירת סקריפט התקנה למכונה"""

        # קריאת .env מהפרויקט הראשי
        env_paths = ["../.env", "../../.env", ".env"]
        env_content = ""

        for env_path in env_paths:
            if os.path.exists(env_path):
                with open(env_path, 'r', encoding='utf-8') as f:
                    env_content = f.read()
                print(f"✅ נמצא .env ב-{env_path}")
                break

        if not env_content:
            print("⚠️ לא נמצא קובץ .env - המכונות עלולות לא לעבוד ללא Google API Key")

        # קריאת AWS credentials
        aws_creds_path = os.path.expanduser("~/.aws/credentials")
        aws_config_path = os.path.expanduser("~/.aws/config")

        aws_creds = ""
        aws_config = ""

        try:
            with open(aws_creds_path, 'r') as f:
                aws_creds = f.read()
        except:
            pass

        try:
            with open(aws_config_path, 'r') as f:
                aws_config = f.read()
        except:
            pass

        # קריאת worker.py
        with open('worker.py', 'r', encoding='utf-8') as f:
            worker_content = f.read()

        # יצירת user data script מעודכן לUbuntu 24.04
        user_data = """#!/bin/bash

# לוג הכל
exec > >(tee /var/log/user-data.log) 2>&1
echo "=== Starting User Data ==="
date

# עדכון מערכת
echo "=== Updating system ==="
apt-get update -y

# התקנת Python והכלים הבסיסיים
echo "=== Installing Python and tools ==="
apt-get install -y python3 python3-pip python3-venv python3-full curl unzip

# התקנת AWS CLI v2
echo "=== Installing AWS CLI v2 ==="
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
rm -rf aws awscliv2.zip

# יצירת virtual environment
echo "=== Creating virtual environment ==="
python3 -m venv /opt/venv
source /opt/venv/bin/activate

# ודא שהכל מותקן
echo "=== Verifying installations ==="
python3 --version
pip --version
/usr/local/bin/aws --version

# יצירת .env
echo "=== Creating .env ==="
cat > .env << 'EOF'
""" + env_content + """
EOF

# הגדרת AWS credentials
echo "=== Setting up AWS credentials ==="
mkdir -p ~/.aws
cat > ~/.aws/credentials << 'EOF'
""" + aws_creds + """
EOF

cat > ~/.aws/config << 'EOF'
""" + aws_config + """
EOF

# הגדרת AWS environment
export AWS_DEFAULT_REGION=""" + REGION + """

# התקנת Python packages
echo "=== Installing Python packages ==="
pip install google-generativeai boto3 pandas python-dotenv

# העתקת worker script
echo "=== Creating worker script ==="
cat > worker.py << 'EOF'
""" + worker_content + """
EOF

# הרצת Worker עם virtual environment
echo "=== Running Worker ==="
source /opt/venv/bin/activate
python3 worker.py --prefix """ + task['prefix'] + """ --part """ + str(task['part_number']) + """ --dataset """ + task[
            'dataset_name'] + """

# כיבוי המכונה
echo "=== Shutting down ==="
shutdown -h now
"""

        return user_data

    def create_security_group(self):
        """יצירת Security Group"""
        try:
            response = self.ec2_client.describe_security_groups(
                GroupNames=[SECURITY_GROUP_NAME]
            )
            return response['SecurityGroups'][0]['GroupId']
        except:
            print(f"🔧 יוצר Security Group {SECURITY_GROUP_NAME}")

            response = self.ec2_client.create_security_group(
                GroupName=SECURITY_GROUP_NAME,
                Description='Gepeta Workers Security Group'
            )

            sg_id = response['GroupId']

            # הוספת SSH access
            self.ec2_client.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[{
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }]
            )

            return sg_id

    def launch_single_instance(self, task, security_group_id):
        """הפעלת מכונה יחידה"""
        try:
            user_data = self.create_user_data_script(task)

            response = self.ec2_client.run_instances(
                ImageId='ami-04a81a99f5ec58529',  # Ubuntu 22.04 LTS
                MinCount=1,
                MaxCount=1,
                InstanceType=INSTANCE_TYPE,
                KeyName=KEY_PAIR_NAME,
                SecurityGroupIds=[security_group_id],
                UserData=user_data,
                TagSpecifications=[{
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': f"gepeta-worker-{task['task_id']}"},
                        {'Key': 'Project', 'Value': 'Gepeta'},
                        {'Key': 'TaskID', 'Value': task['task_id']}
                    ]
                }]
            )

            return {
                'instance_id': response['Instances'][0]['InstanceId'],
                'task': task,
                'status': 'launching'
            }

        except Exception as e:
            print(f"❌ שגיאה בהפעלת {task['task_id']}: {e}")
            return None

    def launch_all_instances(self):
        """הפעלת כל המכונות"""
        print(f"🚀 מפעיל {len(self.tasks)} מכונות...")

        security_group_id = self.create_security_group()
        successful_launches = []

        # הפעלה בקבוצות של 20
        batch_size = 20
        for i in range(0, len(self.tasks), batch_size):
            batch = self.tasks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(self.tasks) + batch_size - 1) // batch_size

            print(f"📦 קבוצה {batch_num}/{total_batches} ({len(batch)} מכונות)")

            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_task = {
                    executor.submit(self.launch_single_instance, task, security_group_id): task
                    for task in batch
                }

                for future in as_completed(future_to_task):
                    result = future.result()
                    if result:
                        successful_launches.append(result)
                        print(f"  ✅ {result['task']['task_id']}")

            if batch_num < total_batches:
                print("⏳ המתנה 10 שניות...")
                time.sleep(10)

        self.instances = successful_launches
        print(f"🎉 הופעלו {len(successful_launches)} מכונות")
        return len(successful_launches) > 0

    def monitor_progress(self):
        """מעקב התקדמות בזמן אמת"""
        print(f"\n📊 מעקב אחר {len(self.instances)} מכונות...")

        while True:
            try:
                statuses = {}

                for instance in self.instances:
                    task_id = instance['task']['task_id']
                    try:
                        status_key = f"{STATUS_PREFIX}{task_id}.json"
                        response = self.s3_client.get_object(
                            Bucket=STATUS_BUCKET,
                            Key=status_key
                        )
                        status_data = json.loads(response['Body'].read().decode('utf-8'))
                        statuses[task_id] = status_data
                    except:
                        statuses[task_id] = {
                            'status': 'starting',
                            'progress_percent': 0,
                            'worker_id': task_id,
                            'prefix': instance['task']['prefix'],
                            'part_number': instance['task']['part_number']
                        }

                # ניקוי מסך והצגה
                os.system('clear' if os.name == 'posix' else 'cls')

                print(f"🎯 Gepeta Progress Monitor - {datetime.now().strftime('%H:%M:%S')}")
                print("=" * 80)

                completed = processing = errors = starting = 0

                for task_id, status in sorted(statuses.items()):
                    worker_status = status.get('status', 'unknown')
                    progress = status.get('progress_percent', 0)
                    prefix = status.get('prefix', 'unknown')
                    part = status.get('part_number', '?')

                    if worker_status == 'completed':
                        symbol = "✅"
                        completed += 1
                    elif worker_status == 'error':
                        symbol = "❌"
                        errors += 1
                    elif worker_status in ['processing', 'loading_file', 'saving']:
                        symbol = "🔄"
                        processing += 1
                    else:
                        symbol = "⏳"
                        starting += 1

                    bar_length = 15
                    filled = int(bar_length * progress / 100)
                    bar = "█" * filled + "░" * (bar_length - filled)

                    print(f"{symbol} {prefix:12} part-{part:2} │{bar}│ {progress:5.1f}%")

                print("=" * 80)
                print(f"📊 ✅ {completed} | 🔄 {processing} | ⏳ {starting} | ❌ {errors}")

                if completed == len(self.instances):
                    print("🎉 כל המכונות הושלמו!")
                    break

                if completed + errors == len(self.instances):
                    print(f"⚠️ כל המכונות הסתיימו. {errors} כשלונות.")
                    break

                time.sleep(5)

            except KeyboardInterrupt:
                print("\n🛑 מעקב הופסק")
                break

    def generate_summary_reports(self):
        """יצירת דוחות סיכום מפורטים"""
        print("\n📊 יוצר דוחות סיכום...")

        try:
            # איסוף כל נתוני הסטטוס
            all_data = []

            for instance in self.instances:
                task_id = instance['task']['task_id']
                try:
                    status_key = f"{STATUS_PREFIX}{task_id}.json"
                    response = self.s3_client.get_object(
                        Bucket=STATUS_BUCKET,
                        Key=status_key
                    )
                    status_data = json.loads(response['Body'].read().decode('utf-8'))

                    all_data.append({
                        'Dataset': status_data.get('dataset', instance['task']['dataset_name']),
                        'Prefix': status_data.get('prefix', instance['task']['prefix']),
                        'Part': status_data.get('part_number', instance['task']['part_number']),
                        'Status': status_data.get('status', 'unknown'),
                        'Original_Words': status_data.get('total_original_words', 0),
                        'Cleaned_Words': status_data.get('total_cleaned_words', 0),
                        'Instance_ID': instance['instance_id'],
                        'Task_ID': task_id,
                        'Target_Path': status_data.get('target_path', 'unknown')
                    })

                except Exception as e:
                    # אם אין נתוני סטטוס
                    all_data.append({
                        'Dataset': instance['task']['dataset_name'],
                        'Prefix': instance['task']['prefix'],
                        'Part': instance['task']['part_number'],
                        'Status': 'no_status',
                        'Original_Words': 0,
                        'Cleaned_Words': 0,
                        'Instance_ID': instance['instance_id'],
                        'Task_ID': task_id,
                        'Target_Path': 'failed'
                    })

            # יצירת DataFrame מפורט
            df_detailed = pd.DataFrame(all_data)

            # יצירת דוח מסכם לפי dataset
            completed_data = df_detailed[df_detailed['Status'] == 'completed']

            if len(completed_data) > 0:
                summary_by_dataset = completed_data.groupby('Dataset').agg({
                    'Original_Words': 'sum',
                    'Cleaned_Words': 'sum',
                    'Part': 'count'  # מספר חלקים שהושלמו
                }).reset_index()

                summary_by_dataset['Files_Completed'] = summary_by_dataset['Part']
                summary_by_dataset['Reduction_Percent'] = (
                        (summary_by_dataset['Original_Words'] - summary_by_dataset['Cleaned_Words']) /
                        summary_by_dataset['Original_Words'] * 100
                ).round(1)

                # שמירת הדוחות עם timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # דוח מפורט
                detailed_filename = f"gepeta_detailed_report_{timestamp}.csv"
                df_detailed.to_csv(detailed_filename, index=False, encoding='utf-8-sig')

                # דוח מסכם
                summary_filename = f"gepeta_summary_report_{timestamp}.csv"
                summary_by_dataset.to_csv(summary_filename, index=False, encoding='utf-8-sig')

                # שמירה ל-S3
                try:
                    # העלאת דוח מפורט ל-S3
                    detailed_key = f"reports/detailed_report_{timestamp}.csv"
                    with open(detailed_filename, 'rb') as f:
                        self.s3_client.put_object(
                            Bucket=STATUS_BUCKET,
                            Key=detailed_key,
                            Body=f.read(),
                            ContentType='text/csv'
                        )

                    # העלאת דוח מסכם ל-S3
                    summary_key = f"reports/summary_report_{timestamp}.csv"
                    with open(summary_filename, 'rb') as f:
                        self.s3_client.put_object(
                            Bucket=STATUS_BUCKET,
                            Key=summary_key,
                            Body=f.read(),
                            ContentType='text/csv'
                        )

                    print(f"☁️ דוחות נשמרו ב-S3:")
                    print(f"   • s3://{STATUS_BUCKET}/{detailed_key}")
                    print(f"   • s3://{STATUS_BUCKET}/{summary_key}")

                except Exception as e:
                    print(f"⚠️ שגיאה בהעלאה ל-S3: {e}")

                print(f"📁 דוחות מקומיים:")
                print(f"   • {detailed_filename}")
                print(f"   • {summary_filename}")

                # הצגת סיכום מהיר
                print(f"\n📈 סיכום מהיר:")
                print("=" * 60)

                total_completed = len(completed_data)
                total_failed = len(df_detailed[df_detailed['Status'] == 'error'])
                total_no_status = len(df_detailed[df_detailed['Status'] == 'no_status'])

                print(f"📊 סטטוס כללי:")
                print(f"   ✅ הושלמו בהצלחה: {total_completed}")
                print(f"   ❌ כשלונות: {total_failed}")
                print(f"   ⚠️ ללא סטטוס: {total_no_status}")
                print(f"   📁 סה\"כ: {len(df_detailed)}")

                if len(summary_by_dataset) > 0:
                    print(f"\n📝 סיכום מילים לפי dataset:")
                    for _, row in summary_by_dataset.iterrows():
                        original = int(row['Original_Words'])
                        cleaned = int(row['Cleaned_Words'])
                        reduction = row['Reduction_Percent']
                        files = int(row['Files_Completed'])

                        print(
                            f"   • {row['Dataset']:12}: {files:2} קבצים | {original:,} → {cleaned:,} מילים ({reduction}% הפחתה)")

                    # סיכום כולל
                    total_original = summary_by_dataset['Original_Words'].sum()
                    total_cleaned = summary_by_dataset['Cleaned_Words'].sum()
                    total_reduction = (
                                (total_original - total_cleaned) / total_original * 100) if total_original > 0 else 0

                    print("=" * 60)
                    print(f"🎯 סה\"כ כולל: {total_original:,} → {total_cleaned:,} מילים ({total_reduction:.1f}% הפחתה)")

                return detailed_filename, summary_filename

            else:
                print("⚠️ אין נתונים מושלמים ליצירת דוח")
                return None, None

        except Exception as e:
            print(f"❌ שגיאה ביצירת דוחות: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def cleanup_instances(self):
        """ניקוי מכונות"""
        if not self.instances:
            return

        instance_ids = [inst['instance_id'] for inst in self.instances]

        try:
            print(f"🧹 מסיים {len(instance_ids)} מכונות...")
            self.ec2_client.terminate_instances(InstanceIds=instance_ids)
            print("✅ פקודת סיום נשלחה")
        except Exception as e:
            print(f"⚠️ שגיאה בניקוי: {e}")


def main():
    print("🚀 Gepeta EC2 Orchestrator")
    print("=" * 40)

    try:
        orchestrator = GepetaOrchestrator()

        # טעינת משימות
        tasks = orchestrator.load_datasets_data()
        if not tasks:
            print("❌ לא נמצאו משימות")
            return

        # אישור משתמש
        estimated_cost = len(tasks) * 0.085 * 2
        print(f"\n💰 עלות משוערת: ~${estimated_cost:.2f}")

        response = input(f"להפעיל {len(tasks)} מכונות? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ בוטל")
            return

        # הפעלה
        success = orchestrator.launch_all_instances()
        if not success:
            print("❌ כשל בהפעלה")
            return

        print("⏳ מתחיל מעקב בעוד 30 שניות...")
        time.sleep(30)

        orchestrator.monitor_progress()

        # יצירת דוחות סיכום
        print("\n" + "=" * 60)
        detailed_report, summary_report = orchestrator.generate_summary_reports()

        # שאלה על ניקוי
        response = input("\n🧹 לסיים ולנקות מכונות? (y/N): ").strip().lower()
        if response == 'y':
            orchestrator.cleanup_instances()
        else:
            print("⚠️ המכונות נשארות פעילות - זכור לסיים אותן ידנית!")

        if detailed_report and summary_report:
            print(f"\n📊 דוחות נוצרו:")
            print(f"   📋 מפורט: {detailed_report}")
            print(f"   📈 מסכם: {summary_report}")

        print("🎉 הושלם!")

    except Exception as e:
        print(f"❌ שגיאה: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()