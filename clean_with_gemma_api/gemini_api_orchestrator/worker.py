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
            'start_time': datetime.now().isoformat(),
            'total_rows': 0,
            'rows_already_clean': 0,
            'rows_processed_now': 0,
            'rate_limit_errors_found': 0,
            'rows_skipped_so_far': 0,  # כמה כבר דילגנו
            'rows_processed_so_far': 0,  # כמה כבר עיבדנו
            'new_rate_limit_errors': 0  # שגיאות rate limit חדשות שיצרנו
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
        """בדיקה אם cleaned_text תקין (לא rate limit error)"""
        if pd.isna(text) or not isinstance(text, str):
            return False

        # אם זה שגיאה, לא תקין
        if text.startswith("[API_ERROR]") or text.startswith("[RATE_LIMIT_ERROR]"):
            return False

        # אם זה ריק או קצר מדי, לא תקין
        if len(text.strip()) < 3:
            return False

        return True

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
            # אם זה שגיאת rate limit - עדכן מונה
            error_msg = f"[API_ERROR] {str(e)}"
            if "429" in str(e) or "RATE_LIMIT_EXCEEDED" in str(e) or "Quota exceeded" in str(e):
                self.stats['new_rate_limit_errors'] += 1
            return error_msg

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
                self.stats['rows_processed_now'] = 0  # יעודכן במהלך העיבוד
                self.stats['rate_limit_errors_found'] = rate_limit_errors

                print(f"📈 ניתוח קובץ מעובד:")
                print(f"   • סה\"כ שורות: {total_rows:,}")
                print(f"   • כבר נקיות: {valid_clean:,}")
                print(f"   • שגיאות rate limit: {rate_limit_errors:,}")

                if rate_limit_errors > 0:
                    # עיבוד רק שורות עם rate limit errors
                    texts_to_process = []
                    indices_to_process = []

                    for idx, row in df.iterrows():
                        if self.is_rate_limit_error(row.get('cleaned_text')):
                            text = row['text']
                            if pd.notna(text) and len(str(text).strip()) > 0:
                                texts_to_process.append(str(text))
                                indices_to_process.append(idx)

                    print(f"🔄 מתקן {len(texts_to_process)} שגיאות rate limit...")

                    # עדכון מספר השורות שמתעבדות
                    self.stats['rows_processed_now'] = len(texts_to_process)

                    # שלח עדכון סטטוס עם כל הנתונים
                    self.update_status("processing")

                    if texts_to_process:
                        total_batches = (len(texts_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
                        self.update_status("processing")

                        processed_results = []
                        for batch_idx in range(0, len(texts_to_process), BATCH_SIZE):
                            batch_num = (batch_idx // BATCH_SIZE) + 1
                            batch = texts_to_process[batch_idx:batch_idx + BATCH_SIZE]

                            progress_percent = (batch_num / total_batches) * 100
                            self.update_status("processing",
                                               current_batch=batch_num,
                                               progress_percent=progress_percent)

                            cleaned_batch = self.process_texts_parallel(batch)
                            processed_results.extend(cleaned_batch)

                            # עדכון מונה השורות שעובדו
                            self.stats['rows_processed_so_far'] = min(len(processed_results), len(texts_to_process))
                            self.update_status("processing", progress_percent=progress_percent)

                            if batch_num < total_batches:
                                time.sleep(0.5)

                        # החלפת שגיאות rate limit בטקסט נקי
                        for i, idx in enumerate(indices_to_process):
                            if i < len(processed_results):
                                df.loc[idx, 'cleaned_text'] = processed_results[i]

                    # עכשיו עבור על כל השורות לעדכון מונים סופי
                    print("📊 מעדכן מונים...")
                    for idx, row in df.iterrows():
                        if self.is_valid_clean_text(row.get('cleaned_text')):
                            # שורה נקייה - נחשבת כ"דולגה"
                            if idx < valid_clean:  # רק אם באמת היתה נקייה מלכתחילה
                                self.stats['rows_skipped_so_far'] = min(self.stats['rows_skipped_so_far'] + 1,
                                                                        valid_clean)

                        # עדכן סטטוס כל 500 שורות
                        if (idx + 1) % 500 == 0:
                            self.update_status("processing")

                    # סיימנו - עדכן לסטטוס סופי
                    self.stats['rows_skipped_so_far'] = valid_clean

                else:
                    print("✅ כל הטקסטים כבר נקיים - רק סופר מילים")
                    # כל השורות נקיות - עדכן מונה הדילוגים
                    self.stats['rows_skipped_so_far'] = total_rows

                df_result = df.copy()
            else:
                # קובץ מקורי - עיבוד מלא כמו הקוד המקורי
                print("🔄 קובץ מקורי - מעבד הכל")

                texts = df['text'].dropna().tolist()
                if not texts:
                    raise ValueError(f"אין טקסטים ב-{filename}")

                # עדכון stats לעיבוד מלא
                self.stats['total_rows'] = len(texts)
                self.stats['rows_already_clean'] = 0
                self.stats['rows_processed_now'] = len(texts)
                self.stats['rate_limit_errors_found'] = 0

                total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
                self.update_status("processing")

                all_cleaned_texts = []

                for batch_idx in range(0, len(texts), BATCH_SIZE):
                    batch_num = (batch_idx // BATCH_SIZE) + 1
                    batch = texts[batch_idx:batch_idx + BATCH_SIZE]

                    progress_percent = (batch_num / total_batches) * 100

                    # עדכן מונה השורות שעובדו עד כה
                    self.stats['rows_processed_so_far'] = min(batch_idx + len(batch), len(texts))

                    self.update_status("processing",
                                       current_batch=batch_num,
                                       progress_percent=progress_percent)

                    cleaned_batch = self.process_texts_parallel(batch)
                    all_cleaned_texts.extend(cleaned_batch)

                    if batch_num < total_batches:
                        time.sleep(0.5)

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


def get_api_key_for_worker(part_number):
    """בחירת API Key לפי מספר ה-part"""
    # חלוקה של 143 מכונות בין 2 API Keys
    # מכונות 0-70: SANDBOX_1 (71 מכונות)
    # מכונות 71-142: SANDBOX_2 (72 מכונות)
    if part_number <= 70:
        api_key = os.getenv("GOOGLE_API_KEY_SANDBOX_1")
        key_name = "SANDBOX_1"
    else:
        api_key = os.getenv("GOOGLE_API_KEY_SANDBOX_2")
        key_name = "SANDBOX_2"

    if not api_key:
        # fallback לSANDBOX_2 אם המפתח לא נמצא
        api_key = os.getenv("GOOGLE_API_KEY_SANDBOX_2")
        key_name = "SANDBOX_2_FALLBACK"

    print(f"🔑 משתמש ב-API Key: {key_name} (part-{part_number})")
    return api_key


def main():
    parser = argparse.ArgumentParser(description='Gepeta Single File Worker')
    parser.add_argument('--prefix', required=True, help='Dataset prefix')
    parser.add_argument('--part', type=int, required=True, help='Part number')
    parser.add_argument('--dataset', required=True, help='Dataset name')

    args = parser.parse_args()

    # בחירת API Key לפי part number
    api_key = get_api_key_for_worker(args.part)
    if not api_key:
        print("❌ לא נמצא API Key מתאים")
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