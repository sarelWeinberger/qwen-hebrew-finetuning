#!/usr/bin/env python3
"""
Gepeta EC2 Worker - עובד יחיד לקובץ אחד
Usage: python worker.py --prefix Geektime --part 0 --dataset geektime

NOTE: This script expects to run in a virtual environment at /opt/venv
"""

import google.generativeai as genai
import boto3
import pandas as pd
from io import StringIO
import time
import os
import argparse
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# הגדרות S3
SOURCE_BUCKET = "gepeta-datasets"
SOURCE_PREFIX = "partly-processed/regex-and-dedup"
TARGET_BUCKET = "gepeta-datasets"
TARGET_PREFIX = "processed/"
STATUS_BUCKET = "gepeta-datasets"
STATUS_PREFIX = "worker-status/"

# הגדרות עיבוד
MAX_WORKERS = 10
BATCH_SIZE = 50


class SingleFileProcessor:
    def __init__(self, api_key, prefix, part_number, dataset_name):
        if not api_key:
            raise ValueError("❌ חסר GOOGLE_API_KEY!")

        self.prefix = prefix
        self.part_number = part_number
        self.dataset_name = dataset_name.lower()

        genai.configure(api_key=api_key)
        self.model_name = 'gemini-2.0-flash'
        self.s3_client = boto3.client('s3')
        self.worker_id = f"{prefix}_part-{part_number}"

        self.stats = {
            'worker_id': self.worker_id,
            'prefix': prefix,
            'part_number': part_number,
            'dataset': dataset_name,
            'status': 'starting',
            'progress_percent': 0,
            'start_time': datetime.now().isoformat()
        }

    def update_status(self, status, **kwargs):
        """עדכון סטטוס ודיווח"""
        self.stats['status'] = status
        self.stats['last_update'] = datetime.now().isoformat()

        for key, value in kwargs.items():
            self.stats[key] = value

        try:
            status_key = f"{STATUS_PREFIX}{self.worker_id}.json"
            self.s3_client.put_object(
                Bucket=STATUS_BUCKET,
                Key=status_key,
                Body=json.dumps(self.stats, ensure_ascii=False, indent=2),
                ContentType='application/json'
            )
        except Exception as e:
            print(f"⚠️ שגיאה בעדכון סטטוס: {e}")

        percent = self.stats.get('progress_percent', 0)
        print(f"📊 {self.worker_id}: {status} - {percent:.1f}%")

    def find_target_file(self):
        """מציאת הקובץ המתאים"""
        self.update_status("searching_file")

        paginator = self.s3_client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=SOURCE_BUCKET, Prefix=SOURCE_PREFIX):
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']
                filename = os.path.basename(key)

                if (filename.startswith(self.prefix) and
                        f"part-{self.part_number}" in filename and
                        filename.endswith('.csv') and
                        obj['Size'] > 0):
                    print(f"✅ נמצא קובץ: {filename}")
                    return key

        raise FileNotFoundError(f"לא נמצא קובץ עבור {self.prefix} part-{self.part_number}")

    def clean_text_with_api(self, text):
        """ניקוי טקסט עם Google API"""
        model = genai.GenerativeModel(self.model_name)

        prompt = f"""נקה את הטקסט העברי הבא מפגמי קידוד, תגיות HTML, פרסומות ותבניות. החזר רק טקסט נקי בעברית:

{text}

טקסט נקי:"""

        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"[API_ERROR] {str(e)}"

    def process_texts_parallel(self, texts):
        """עיבוד מקבילי"""
        results = [''] * len(texts)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_index = {
                executor.submit(self.clean_text_with_api, text): i
                for i, text in enumerate(texts)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    results[index] = f"[ERROR] {str(e)}"

        return results

    def process_file(self):
        """עיבוד הקובץ הראשי"""
        try:
            # חיפוש וטעינת קובץ
            file_key = self.find_target_file()
            filename = os.path.basename(file_key)

            self.update_status("loading_file")

            response = self.s3_client.get_object(Bucket=SOURCE_BUCKET, Key=file_key)
            content = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(content))

            print(f"📊 נטען: {len(df)} שורות")

            if 'text' not in df.columns or 'n_count' not in df.columns:
                raise ValueError(f"חסרות עמודות נדרשות ב-{filename}")

            # עיבוד
            texts = df['text'].dropna().tolist()
            if not texts:
                raise ValueError(f"אין טקסטים ב-{filename}")

            total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
            self.update_status("processing", total_batches=total_batches)

            all_cleaned_texts = []

            for batch_idx in range(0, len(texts), BATCH_SIZE):
                batch_num = (batch_idx // BATCH_SIZE) + 1
                batch = texts[batch_idx:batch_idx + BATCH_SIZE]

                progress_percent = (batch_num / total_batches) * 100
                self.update_status("processing",
                                   current_batch=batch_num,
                                   progress_percent=progress_percent)

                cleaned_batch = self.process_texts_parallel(batch)
                all_cleaned_texts.extend(cleaned_batch)

                if batch_num < total_batches:
                    time.sleep(0.5)

            # שמירה
            df_result = df.copy()
            df_result['cleaned_text'] = all_cleaned_texts[:len(df)]

            original_words = (df_result['n_count'].sum() - 1)
            cleaned_words = df_result['cleaned_text'].apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            ).sum()

            self.update_status("saving")

            target_key = f"{TARGET_PREFIX}{self.dataset_name}/{filename}"

            csv_buffer = StringIO()
            df_result.to_csv(csv_buffer, index=False, encoding='utf-8')

            self.s3_client.put_object(
                Bucket=TARGET_BUCKET,
                Key=target_key,
                Body=csv_buffer.getvalue(),
                ContentType='text/csv'
            )

            self.update_status("completed",
                               progress_percent=100,
                               total_original_words=original_words,
                               total_cleaned_words=cleaned_words,
                               target_path=f"s3://{TARGET_BUCKET}/{target_key}")

            print(f"✅ הושלם: {self.worker_id}")
            print(f"💾 נשמר ב: s3://{TARGET_BUCKET}/{target_key}")
            return True

        except Exception as e:
            self.update_status("error", error_message=str(e))
            print(f"❌ שגיאה ב-{self.worker_id}: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Gepeta Single File Worker')
    parser.add_argument('--prefix', required=True, help='Dataset prefix')
    parser.add_argument('--part', type=int, required=True, help='Part number')
    parser.add_argument('--dataset', required=True, help='Dataset name')

    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY_SANDBOX_2")
    if not api_key:
        print("❌ חסר GOOGLE_API_KEY_SANDBOX_2")
        return False

    try:
        processor = SingleFileProcessor(
            api_key=api_key,
            prefix=args.prefix,
            part_number=args.part,
            dataset_name=args.dataset
        )

        success = processor.process_file()
        return success

    except Exception as e:
        print(f"❌ שגיאה קריטית: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)