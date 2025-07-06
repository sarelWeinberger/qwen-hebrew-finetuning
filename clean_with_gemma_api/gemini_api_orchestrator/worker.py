#!/usr/bin/env python3
"""
Gepeta EC2 Worker - עובד יחיד לקובץ אחד עם שימוש בבאצ'ים
Usage: python worker.py --prefix Geektime --part 0 --dataset geektime

NOTE: This script expects to run in a virtual environment at /opt/venv
"""

from google import genai
from google.genai.types import CreateBatchJobConfig
import boto3
import pandas as pd
from io import StringIO
import time
import os
import argparse
import json
from datetime import datetime
import jsonlines
import fsspec
from dotenv import load_dotenv

load_dotenv()

# הגדרות S3
SOURCE_BUCKET = "gepeta-datasets"
SOURCE_PREFIX = "partly-processed/regex-and-dedup"
TARGET_BUCKET = "gepeta-datasets"
TARGET_PREFIX = "processed/"
STATUS_BUCKET = "gepeta-datasets"
STATUS_PREFIX = "worker-status/"

# הגדרות GCP
PROJECT_ID = "pwcnext-sandbox01"
LOCATION = "us-central1"
BATCH_BUCKET = "gepeta-batches"

# הגדרות מודל
MODEL_ID = "gemini-2.0-flash-001"


class SingleFileProcessor:
    def __init__(self, api_key, prefix, part_number, dataset_name):
        self.prefix = prefix
        self.part_number = part_number
        self.dataset_name = dataset_name.lower()

        # Initialize GCP Vertex AI client
        self.client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

        self.s3_client = boto3.client('s3')
        self.worker_id = f"{prefix}_part-{part_number}"

        self.stats = {
            'worker_id': self.worker_id,
            'prefix': prefix,
            'part_number': part_number,
            'dataset': dataset_name,
            'status': 'starting',
            'progress_percent': 0,
            'start_time': datetime.now().isoformat(),
            'total_rows': 0,
            'rows_already_clean': 0,
            'rows_processed_now': 0,
            'rate_limit_errors_found': 0,
            'rows_skipped_so_far': 0,
            'rows_processed_so_far': 0,
            'new_rate_limit_errors': 0
        }

    def update_status(self, status, **kwargs):
        """עדכון סטטוס ודיווח"""
        self.stats['status'] = status
        self.stats['last_update'] = datetime.now().isoformat()

        for key, value in kwargs.items():
            self.stats[key] = value

        try:
            status_key = f"{STATUS_PREFIX}{self.worker_id}.json"

            # המרה של numpy types ל-Python types לפני JSON
            safe_stats = {}
            for key, value in self.stats.items():
                if hasattr(value, 'item'):  # numpy scalar
                    safe_stats[key] = value.item()
                elif isinstance(value, (int, float)):
                    safe_stats[key] = int(value) if isinstance(value, int) else float(value)
                else:
                    safe_stats[key] = value

            self.s3_client.put_object(
                Bucket=STATUS_BUCKET,
                Key=status_key,
                Body=json.dumps(safe_stats, ensure_ascii=False, indent=2),
                ContentType='application/json'
            )
        except Exception as e:
            print(f"⚠️ שגיאה בעדכון סטטוס: {e}")

        percent = self.stats.get('progress_percent', 0)
        print(f"📊 {self.worker_id}: {status} - {percent:.1f}%")

    def find_target_file(self):
        """מציאת הקובץ המעובד ב-processed/"""
        self.update_status("searching_file")

        # חיפוש ישר בתיקיית processed
        processed_prefix = f"{TARGET_PREFIX}{self.dataset_name}/"

        print(f"🔍 מחפש קובץ מעובד ב-{processed_prefix}")

        paginator = self.s3_client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=TARGET_BUCKET, Prefix=processed_prefix):
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']
                filename = os.path.basename(key)

                if (filename.startswith(self.prefix) and
                        f"part-{self.part_number}" in filename and
                        filename.endswith('.csv') and
                        obj['Size'] > 0):
                    print(f"✅ נמצא קובץ מעובד: {filename}")
                    return TARGET_BUCKET, key

        # אם לא נמצא בprocessed, חפש במקום המקורי
        print(f"⚠️ לא נמצא קובץ מעובד, מחפש במקור...")

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
                    print(f"✅ נמצא קובץ מקורי: {filename}")
                    return SOURCE_BUCKET, key

        raise FileNotFoundError(f"לא נמצא קובץ עבור {self.prefix} part-{self.part_number}")

    def count_words(self, text):
        """ספירת מילים בטקסט"""
        if pd.isna(text) or not isinstance(text, str):
            return 0
        return len(str(text).split())

    def is_valid_clean_text(self, text):
        """בדיקה אם cleaned_text תקין (לא rate limit error)"""
        if pd.isna(text) or not isinstance(text, str):
            return False

        # Debug print
        if text.startswith("[API_ERROR]"):
            print(f"🔍 DEBUG: נמצא API_ERROR: {text[:100]}...")
            return False

        # אם זה ריק או קצר מדי, לא תקין
        if len(text.strip()) < 3:
            print(f"🔍 DEBUG: טקסט קצר מדי: '{text}'")
            return False

        return True

    def is_rate_limit_error(self, text):
        """בדיקה אם הטקסט הוא שגיאת rate limit"""
        if not isinstance(text, str):
            return False

        is_error = (text.startswith("[API_ERROR] 429") or
                    "RATE_LIMIT_EXCEEDED" in text or
                    "Quota exceeded" in text)

        if is_error:
            print(f"🔍 DEBUG: נמצא Rate Limit Error: {text[:100]}...")

        return is_error

    def create_batch_jsonl(self, texts):
        """יצירת קובץ JSONL עבור הבאצ'"""
        jsonl_filename = f"{self.worker_id}_batch.jsonl"

        prompt = """נקה את הטקסט העברי הבא מפגמי קידוד, תגיות HTML, פרסומות ותבניות. החזר רק טקסט נקי בעברית:

{text}

טקסט נקי:"""

        with jsonlines.open(jsonl_filename, mode="w") as writer:
            for i, text in enumerate(texts):
                generationConfig = {
                    "temperature": 0,
                    "maxOutputTokens": 8192
                }

                writer.write({
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [
                                    {"text": prompt.format(text=text)}
                                ]
                            }
                        ],
                        "generationConfig": generationConfig
                    }
                })

        return jsonl_filename

    def upload_batch_to_gcs(self, jsonl_filename):
        """העלאת קובץ הבאצ' ל-GCS"""
        from google.cloud import storage

        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BATCH_BUCKET)
        blob = bucket.blob(jsonl_filename)
        blob.upload_from_filename(jsonl_filename)

        return f"gs://{BATCH_BUCKET}/{jsonl_filename}"

    def process_texts_with_batch(self, texts):
        """עיבוד טקסטים עם batch API"""
        print(f"🔄 מעבד {len(texts)} טקסטים עם batch API...")

        # יצירת JSONL
        jsonl_filename = self.create_batch_jsonl(texts)

        # העלאה ל-GCS
        input_uri = self.upload_batch_to_gcs(jsonl_filename)

        # יצירת batch job
        dest_uri = f"gs://{BATCH_BUCKET}/results/{self.worker_id}"

        batch_job = self.client.batches.create(
            model=MODEL_ID,
            src=input_uri,
            config=CreateBatchJobConfig(dest=dest_uri),
        )

        print(f"📋 Batch job נוצר: {batch_job.name}")

        # המתנה לסיום
        while not batch_job.state in ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_UNEXECUTED"]:
            time.sleep(10)
            batch_job = self.client.batches.get(name=batch_job.name)
            print(f"⏳ מחכה לסיום batch... סטטוס: {batch_job.state}")

        if batch_job.state == "JOB_STATE_SUCCEEDED":
            print("✅ Batch job הושלם בהצלחה!")

            # קריאת תוצאות
            results = self.load_batch_results(batch_job.dest.gcs_uri)

            # ניקוי
            os.remove(jsonl_filename)
            self.client.batches.delete(name=batch_job.name)

            return results
        else:
            print(f"❌ Batch job נכשל: {batch_job.error}")
            return [f"[API_ERROR] Batch failed: {batch_job.error}"] * len(texts)

    def load_batch_results(self, dest_uri):
        """טעינת תוצאות הבאצ'"""
        fs = fsspec.filesystem("gcs")
        file_paths = fs.glob(f"{dest_uri}/*/predictions.jsonl")

        if not file_paths:
            print("❌ לא נמצאו תוצאות")
            return []

        # טעינת התוצאות
        df = pd.read_json(f"gs://{file_paths[-1]}", lines=True)

        results = []
        for _, row in df.iterrows():
            try:
                response = row['response']
                if 'candidates' in response and response['candidates']:
                    candidate = response['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if parts and 'text' in parts[0]:
                            results.append(parts[0]['text'].strip())
                        else:
                            results.append("[API_ERROR] No text in response")
                    else:
                        results.append("[API_ERROR] No content in candidate")
                else:
                    results.append("[API_ERROR] No candidates in response")
            except Exception as e:
                results.append(f"[API_ERROR] {str(e)}")

        return results

    def process_file(self):
        """עיבוד הקובץ הראשי - עובד על קבצים מעובדים"""
        try:
            # חיפוש וטעינת קובץ (מעובד או מקורי)
            source_bucket, file_key = self.find_target_file()
            filename = os.path.basename(file_key)
            is_processed_file = source_bucket == TARGET_BUCKET

            self.update_status("loading_file")

            response = self.s3_client.get_object(Bucket=source_bucket, Key=file_key)
            content = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(content))

            print(f"📊 נטען: {len(df)} שורות מ-{'processed' if is_processed_file else 'source'}")

            if 'text' not in df.columns or 'n_count' not in df.columns:
                raise ValueError(f"חסרות עמודות נדרשות ב-{filename}")

            # בדיקה אם יש עמודת cleaned_text
            has_cleaned_text = 'cleaned_text' in df.columns
            print(f"📋 יש עמודת cleaned_text: {'✅' if has_cleaned_text else '❌'}")

            if has_cleaned_text:
                # זה קובץ מעובד - נתקן רק rate limit errors
                total_rows = len(df)
                valid_clean = df['cleaned_text'].apply(self.is_valid_clean_text).sum()
                rate_limit_errors = df['cleaned_text'].apply(self.is_rate_limit_error).sum()

                # עדכון stats
                self.stats['total_rows'] = total_rows
                self.stats['rows_already_clean'] = valid_clean
                self.stats['rows_processed_now'] = 0
                self.stats['rate_limit_errors_found'] = rate_limit_errors

                print(f"📈 ניתוח קובץ מעובד:")
                print(f"   • סה\"כ שורות: {total_rows:,}")
                print(f"   • כבר נקיות: {valid_clean:,}")
                print(f"   • שגיאות rate limit: {rate_limit_errors:,}")

                if rate_limit_errors > 0:
                    # איסוף טקסטים לעיבוד
                    texts_to_process = []
                    indices_to_process = []

                    for idx, row in df.iterrows():
                        if self.is_rate_limit_error(row.get('cleaned_text')):
                            text = row['text']
                            if pd.notna(text) and len(str(text).strip()) > 0:
                                texts_to_process.append(str(text))
                                indices_to_process.append(idx)

                    print(f"🔄 מתקן {len(texts_to_process)} שגיאות rate limit...")

                    self.stats['rows_processed_now'] = len(texts_to_process)
                    self.update_status("processing")

                    if texts_to_process:
                        # עיבוד עם batch API
                        self.update_status("processing", progress_percent=25)
                        processed_results = self.process_texts_with_batch(texts_to_process)
                        self.update_status("processing", progress_percent=75)

                        # החלפת שגיאות rate limit בטקסט נקי
                        for i, idx in enumerate(indices_to_process):
                            if i < len(processed_results):
                                df.loc[idx, 'cleaned_text'] = processed_results[i]

                        self.stats['rows_processed_so_far'] = len(processed_results)
                        self.update_status("processing", progress_percent=90)

                    self.stats['rows_skipped_so_far'] = valid_clean
                else:
                    print("✅ כל הטקסטים כבר נקיים - רק סופר מילים")
                    self.stats['rows_skipped_so_far'] = total_rows

                df_result = df.copy()
            else:
                # קובץ מקורי - עיבוד מלא
                print("🔄 קובץ מקורי - מעבד הכל")

                texts = df['text'].dropna().tolist()
                if not texts:
                    raise ValueError(f"אין טקסטים ב-{filename}")

                # עדכון stats לעיבוד מלא
                self.stats['total_rows'] = len(texts)
                self.stats['rows_already_clean'] = 0
                self.stats['rows_processed_now'] = len(texts)
                self.stats['rate_limit_errors_found'] = 0

                self.update_status("processing", progress_percent=10)

                # עיבוד עם batch API
                all_cleaned_texts = self.process_texts_with_batch(texts)

                self.stats['rows_processed_so_far'] = len(all_cleaned_texts)
                self.update_status("processing", progress_percent=80)

                df_result = df.copy()
                df_result['cleaned_text'] = all_cleaned_texts[:len(df)]

            # הוספת/עדכון ספירת מילים
            print("📊 מחשב מספר מילים...")
            self.update_status("calculating_words")
            df_result['cleaned_text_words'] = df_result['cleaned_text'].apply(self.count_words)

            # חישוב סטטיסטיקות
            original_words = (df_result['n_count'].sum() - 1)
            cleaned_words = df_result['cleaned_text_words'].sum()

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
                               total_original_words=int(original_words),
                               total_cleaned_words=int(cleaned_words),
                               target_path=f"s3://{TARGET_BUCKET}/{target_key}")

            print(f"✅ הושלם: {self.worker_id}")
            print(f"💾 נשמר ב: s3://{TARGET_BUCKET}/{target_key}")
            print(f"📊 מילים: {original_words:,} → {cleaned_words:,}")
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

    try:
        processor = SingleFileProcessor(
            api_key=None,  # לא נדרש עוד עבור batch API
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