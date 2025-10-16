
        # העלאה ל-ECR
        push_cmd = f"docker push {image_uri}"
        print(f"📤 מעלה ל-ECR: {push_cmd}")
        result = subprocess.run(push_cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ הועלה בהצלחה: {image_uri}")
            return image_uri
        else:
            print(f"❌ שגיאה בהעלאה: {result.stderr}")
            return None

    def create_model(self, image_uri, model_name=None):
        """
        יצירת SageMaker Model
        """
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_name = f"text-cleaning-model-{timestamp}"

        model_config = {
            'ModelName': model_name,
            'ExecutionRoleArn': self.role_arn,
            'PrimaryContainer': {
                'Image': image_uri,
                'Mode': 'SingleModel',
                'Environment': {
                    'HF_TOKEN': self.hf_token or '',
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
                }
            }
        }

        try:
            response = self.sagemaker.create_model(**model_config)
            print(f"✅ נוצר מודל: {model_name}")
            return model_name
        except Exception as e:
            print(f"❌ שגיאה ביצירת מודל: {e}")
            return None

    def upload_input_data(self, texts, bucket_name, key_prefix="text-cleaning-input"):
        """
        העלאת נתוני קלט ל-S3
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_key = f"{key_prefix}/batch_{timestamp}.jsonl"

        # הכנת נתונים בפורמט JSONL (שורה לכל טקסט)
        jsonl_data = []
        for i, text in enumerate(texts):
            jsonl_data.append(json.dumps({
                "texts": [{"index": i, "text": text}]
            }, ensure_ascii=False))

        # העלאה ל-S3
        self.s3.put_object(
            Bucket=bucket_name,
            Key=input_key,
            Body='\n'.join(jsonl_data),
            ContentType='application/jsonl'
        )

        print(f"📤 הועלה לS3: s3://{bucket_name}/{input_key}")
        return f"s3://{bucket_name}/{input_key}"

    def create_batch_transform_job(self,
                                   model_name,
                                   input_s3_uri,
                                   output_s3_path,
                                   job_name=None,
                                   instance_type="ml.g5.12xlarge",
                                   instance_count=1):
        """
        יצירת Batch Transform Job
        """
        if job_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            job_name = f"text-cleaning-job-{timestamp}"

        job_config = {
            'TransformJobName': job_name,
            'ModelName': model_name,
            'TransformInput': {
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': input_s3_uri
                    }
                },
                'ContentType': 'application/jsonl',
                'SplitType': 'Line'
            },
            'TransformOutput': {
                'S3OutputPath': output_s3_path,
                'Accept': 'application/json'
            },
            'TransformResources': {
                'InstanceType': instance_type,
                'InstanceCount': instance_count
            }
        }

        try:
            response = self.sagemaker.create_transform_job(**job_config)
            print(f"🚀 נוצר Batch Transform Job: {job_name}")
            print(f"📊 סטטוס: {response['TransformJobArn']}")
            return job_name
        except Exception as e:
            print(f"❌ שגיאה ביצירת Job: {e}")
            return None

    def wait_for_job_completion(self, job_name, check_interval=60):
        """
        המתנה לסיום הJob
        """
        print(f"⏳ ממתין לסיום Job: {job_name}")

        while True:
            response = self.sagemaker.describe_transform_job(TransformJobName=job_name)
            status = response['TransformJobStatus']

            print(f"📊 סטטוס: {status}")

            if status == 'Completed':
                print("✅ Job הושלם בהצלחה!")
                return True
            elif status == 'Failed':
                print(f"❌ Job נכשל: {response.get('FailureReason', 'Unknown error')}")
                return False
            elif status in ['Stopping', 'Stopped']:
                print(f"🛑 Job הופסק: {status}")
                return False

            time.sleep(check_interval)

    def download_results(self, output_s3_path, local_dir="./results"):
        """
        הורדת תוצאות מS3
        """
        print(f"📥 מוריד תוצאות מ-{output_s3_path}")

        # Parse S3 path
        if output_s3_path.startswith('s3://'):
            bucket_and_key = output_s3_path[5:].split('/', 1)
            bucket_name = bucket_and_key[0]
            prefix = bucket_and_key[1] if len(bucket_and_key) > 1 else ""

        # Create local directory
        os.makedirs(local_dir, exist_ok=True)

        # List and download files
        response = self.s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        if 'Contents' not in response:
            print("❌ לא נמצאו תוצאות")
            return []

        downloaded_files = []
        for obj in response['Contents']:
            key = obj['Key']
            local_file = os.path.join(local_dir, os.path.basename(key))

            self.s3.download_file(bucket_name, key, local_file)
            downloaded_files.append(local_file)
            print(f"✅ הורד: {local_file}")

        return downloaded_files

# דוגמת שימוש
def main():
    # הגדרות - צריך לעדכן!
    REGION = "us-east-1"
    ROLE_ARN = "arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole"  # עדכן!
    HF_TOKEN = "hf_your_token_here"  # עדכן!
    BUCKET_NAME = "your-s3-bucket"  # עדכן!

    # טקסטים לבדיקה
    sample_texts = [
        "דרעי: אין סיבה שניכנס לעימותים בקואליציה סביב חוק הגיוס. תגיות: דרעי חוק גיוס",
        "חדשות ספורט: הפועל ניצחה!!! © כל הזכויות שמורות... Follow @sport_news",
        "כלכלה: עליית המדד... <phone>03-1234567</phone> לפרטים באתר www.example.co.il",
        "פוליטיקה: ישיבת הממשלה החליטה,,, זה זה מידע חשוב!!! _____ עוד באתר"
    ]

    # יצירת מעבד
    processor = SageMakerBatchProcessor(
        region_name=REGION,
        role_arn=ROLE_ARN,
        hf_token=HF_TOKEN
    )

    try:
        # 1. בניית image
        image_uri = processor.build_and_push_image()
        if not image_uri:
            return

        # 2. יצירת מודל
        model_name = processor.create_model(image_uri)
        if not model_name:
            return

        # 3. העלאת נתונים
        input_s3_uri = processor.upload_input_data(sample_texts, BUCKET_NAME)
        output_s3_path = f"s3://{BUCKET_NAME}/text-cleaning-output/"

        # 4. יצירת Job
        job_name = processor.create_batch_transform_job(
            model_name=model_name,
            input_s3_uri=input_s3_uri,
            output_s3_path=output_s3_path,
            instance_type="ml.g5.12xlarge"
        )

        if not job_name:
            return

        # 5. המתנה לסיום
        if processor.wait_for_job_completion(job_name):
            # 6. הורדת תוצאות
            results = processor.download_results(output_s3_path)
            print(f"📁 תוצאות נשמרו ב: {results}")

    except Exception as e:
        print(f"❌ שגיאה כללית: {e}")

if __name__ == "__main__":
    main()