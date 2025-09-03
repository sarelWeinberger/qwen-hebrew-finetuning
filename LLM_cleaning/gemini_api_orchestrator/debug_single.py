#!/usr/bin/env python3
"""
Debug Single EC2 Instance - בדיקת מכונה אחת
"""

import boto3
import os


def create_simple_user_data():
    """סקריפט פשוט לבדיקה"""

    # קריאת .env
    env_paths = ["../.env", "../../.env", ".env"]
    env_content = ""

    for env_path in env_paths:
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                env_content = f.read()
            print(f"✅ נמצא .env ב-{env_path}")
            break

    if not env_content:
        print("❌ לא נמצא קובץ .env!")
        return None

    # קריאת AWS credentials
    aws_creds_path = os.path.expanduser("~/.aws/credentials")
    if not os.path.exists(aws_creds_path):
        print("❌ לא נמצא ~/.aws/credentials!")
        return None

    with open(aws_creds_path, 'r') as f:
        aws_creds = f.read()

    # יצירת הסקריפט בלי f-string
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

# הגדרת AWS environment
export AWS_DEFAULT_REGION=us-east-1

# התקנת Python packages
echo "=== Installing Python packages ==="
pip install google-generativeai boto3 pandas python-dotenv

# בדיקת Google API
echo "=== Testing Google API ==="
python3 -c "
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY_SANDBOX_2')
print(f'API Key found: {bool(api_key)}')
if api_key:
    genai.configure(api_key=api_key)
    print('Google API configured successfully')
else:
    print('No API key found in .env')
"

# בדיקת S3 connection
echo "=== Testing S3 connection ==="
/usr/local/bin/aws sts get-caller-identity
/usr/local/bin/aws s3 ls s3://gepeta-datasets/partly-processed/regex-and-dedup/ | head -5

# יצירת סטטוס debug
echo "=== Creating debug status ==="
python3 -c "
import boto3
import json
from datetime import datetime

s3 = boto3.client('s3')
status = {
    'worker_id': 'DEBUG_TEST',
    'status': 'debug_completed',
    'timestamp': datetime.now().isoformat(),
    'message': 'Debug script completed successfully'
}

s3.put_object(
    Bucket='gepeta-datasets',
    Key='worker-status/DEBUG_TEST.json',
    Body=json.dumps(status, indent=2),
    ContentType='application/json'
)
print('Debug status saved to S3')
"

echo "=== User Data Completed ==="
date
"""

    return user_data

def launch_debug_instance():
    """הפעלת מכונת debug"""

    ec2_client = boto3.client('ec2', region_name='us-east-1')

    user_data = create_simple_user_data()
    if not user_data:
        print("❌ לא יכול ליצור User Data")
        return None

    try:
        print("🚀 מפעיל מכונת debug...")

        response = ec2_client.run_instances(
            ImageId='ami-04a81a99f5ec58529',  # Ubuntu 22.04 LTS ב-us-east-1
            MinCount=1,
            MaxCount=1,
            InstanceType='t3.micro',  # זול יותר לdebug
            KeyName='gepeta-worker-key',
            UserData=user_data,
            TagSpecifications=[{
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': 'Name', 'Value': 'gepeta-debug-test'},
                    {'Key': 'Project', 'Value': 'Gepeta-Debug'}
                ]
            }]
        )

        instance_id = response['Instances'][0]['InstanceId']
        print(f"✅ מכונת debug הופעלה: {instance_id}")

        print("📊 מה לבדוק:")
        print("1. חכה 2-3 דקות")
        print("2. AWS Console → EC2 → בחר את המכונה")
        print("3. Actions → Monitor → Get system log")
        print("4. בדוק S3: gepeta-datasets/worker-status/DEBUG_TEST.json")

        return instance_id

    except Exception as e:
        print(f"❌ שגיאה בהפעלת debug: {e}")
        return None


def check_debug_status():
    """בדיקת סטטוס debug"""
    try:
        s3_client = boto3.client('s3')

        response = s3_client.get_object(
            Bucket='gepeta-datasets',
            Key='worker-status/DEBUG_TEST.json'
        )

        import json
        status = json.loads(response['Body'].read().decode('utf-8'))
        print("✅ Debug הצליח!")
        print(json.dumps(status, indent=2))
        return True

    except Exception as e:
        print(f"❌ Debug לא הצליח: {e}")
        return False


def main():
    print("🔍 Gepeta Debug Tool")
    print("=" * 40)

    choice = input("1. הפעל debug instance\n2. בדוק סטטוס debug\n3. יציאה\nבחר: ")

    if choice == "1":
        launch_debug_instance()
    elif choice == "2":
        check_debug_status()
    else:
        print("👋 יום טוב!")


if __name__ == "__main__":
    main()