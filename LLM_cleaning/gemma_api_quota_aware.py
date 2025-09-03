#!/usr/bin/env python3
"""
Gepeta Project - Geektime Files Processor
עיבוד קבצי Geektime עם Google Gemma API

Usage:
1. pip install google-generativeai boto3 pandas python-dotenv
2. הגדר GOOGLE_API_KEY ו-AWS credentials
3. python geektime_processor.py
"""

import google.generativeai as genai
import boto3
import pandas as pd
from io import StringIO
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import json
import random

# =============================================================================
# הגדרות
# =============================================================================

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY_SANDBOX_2", "YOUR_API_KEY_HERE")

# S3 Settings
SOURCE_BUCKET = "gepeta-datasets"
SOURCE_PREFIX = "partly-processed/regex-and-dedup"
TARGET_BUCKET = "gepeta-datasets"
TARGET_PREFIX = "processed/"

# Processing Settings
PREFIX_FILTER = "Geektime"  # הקידומת שאנחנו מעבדים (ניתן לשינוי)
MAX_WORKERS = 10  # הפחתתי מ-10 ל-5 למניעת quota issues
BATCH_SIZE = 25  # הפחתתי מ-50 ל-25 למניעת quota issues
TEST_LIMIT = 100  # מספר קבצים לבדיקה (None לכל הקבצים)

# Retry Settings for Quota Management
MAX_RETRIES = 5  # מספר מקסימלי של ניסיונות
BASE_RETRY_DELAY = 10  # השהיה בסיסית בשניות
MAX_RETRY_DELAY = 300  # השהיה מקסימלית בשניות
QUOTA_COOLDOWN = 60  # השהיה אחרי quota error בשניות

class GeektimeProcessor:
    """מעבד קבצי Geektime עם Google API"""

    def __init__(self, api_key):
        """אתחול המעבד"""
        if api_key == "YOUR_API_KEY_HERE":
            raise ValueError("❌ עדכן את GOOGLE_API_KEY!")

        # Google AI Setup
        genai.configure(api_key=api_key)
        #self.model_name = 'gemma-3-27b-it'
        self.model_name = 'gemini-1.5-flash'

        # S3 Setup
        self.s3_client = boto3.client('s3')

        # Statistics
        self.stats = {
            'files_processed': 0,
            'texts_processed': 0,
            'total_time': 0,
            'errors': [],
            'quota_errors': 0,
            'retries': 0,
            'start_time': datetime.now()
        }

        print("🚀 Gepeta Generic Prefix Processor מוכן")
        print(f"📁 מקור: s3://{SOURCE_BUCKET}/{SOURCE_PREFIX}")
        print(f"🎯 יעד: s3://{TARGET_BUCKET}/{TARGET_PREFIX}{PREFIX_FILTER.lower()}/")
        print(f"🔍 קידומת: {PREFIX_FILTER}")
        print(f"👥 Workers: {MAX_WORKERS}")
        print(f"⏰ התחלה: {self.stats['start_time'].strftime('%H:%M:%S')}")

    def list_geektime_files(self, limit=None):
        """מציאת כל קבצי Geektime"""
        print(f"🔍 מחפש קבצי {PREFIX_FILTER}...")

        files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')

        page_count = 0
        for page in paginator.paginate(Bucket=SOURCE_BUCKET, Prefix=SOURCE_PREFIX):
            page_count += 1
            print(f"  סורק דף {page_count}...")

            if 'Contents' not in page:
                continue

            page_files = [
                obj['Key'] for obj in page['Contents']
                if obj['Key'].endswith('.csv')
                   and PREFIX_FILTER in os.path.basename(obj['Key'])
                   and obj['Size'] > 0
            ]

            files.extend(page_files)
            print(f"  דף {page_count}: נמצאו {len(page_files)} קבצי {PREFIX_FILTER}")

            # הגבלה לבדיקה
            if limit and len(files) >= limit:
                files = files[:limit]
                break

        print(f"✅ נמצאו {len(files)} קבצי {PREFIX_FILTER} כולל")
        return files

    def read_csv_from_s3(self, bucket, key):
        """קריאת CSV מ-S3"""
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(content))
            return df
        except Exception as e:
            print(f"❌ שגיאה בקריאת {key}: {e}")
            return None

    def save_csv_to_s3(self, df, bucket, key):
        """שמירת CSV ל-S3"""
        try:
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8')

            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=csv_buffer.getvalue(),
                ContentType='text/csv'
            )
            return True
        except Exception as e:
            print(f"❌ שגיאה בשמירת {key}: {e}")
            return False

    def is_quota_error(self, error):
        """בדיקה אם השגיאה קשורה ל-quota"""
        error_str = str(error).lower()
        quota_indicators = [
            'quota',
            'rate limit',
            'too many requests',
            'resource_exhausted',
            'quota_value',
            'retry_delay'
        ]
        return any(indicator in error_str for indicator in quota_indicators)

    def extract_retry_delay(self, error):
        """חילוץ זמן המתנה מהשגיאה אם קיים"""
        try:
            error_str = str(error)
            # חיפוש אחר retry_delay
            if 'retry_delay' in error_str and 'seconds:' in error_str:
                import re
                match = re.search(r'seconds:\s*(\d+)', error_str)
                if match:
                    return int(match.group(1))
        except:
            pass
        return None

    def clean_text_with_api(self, text, retry_count=0):
        """ניקוי טקסט יחיד עם Google API וטיפול בשגיאות quota"""
        model = genai.GenerativeModel(self.model_name)

        prompt = f"""נקה את הטקסט העברי הבא מפגמי קידוד, תגיות HTML, פרסומות ותבניות. החזר רק טקסט נקי בעברית:

{text[:800]}

טקסט נקי:"""

        for attempt in range(MAX_RETRIES):
            try:
                response = model.generate_content(prompt)
                return response.text.strip()

            except Exception as e:
                if self.is_quota_error(e):
                    self.stats['quota_errors'] += 1
                    self.stats['retries'] += 1

                    # חילוץ זמן המתנה מהשגיאה
                    suggested_delay = self.extract_retry_delay(e)

                    if suggested_delay:
                        wait_time = suggested_delay + random.randint(1, 5)  # הוספת jitter
                        print(f"⏳ Quota error - ממתין {wait_time} שניות (הוראה מהשרת)")
                    else:
                        # חישוב exponential backoff
                        wait_time = min(BASE_RETRY_DELAY * (2 ** attempt) + random.randint(1, 10), MAX_RETRY_DELAY)
                        print(f"⏳ Quota error - ממתין {wait_time} שניות (ניסיון {attempt + 1}/{MAX_RETRIES})")

                    time.sleep(wait_time)
                    continue
                else:
                    # שגיאה אחרת - לא quota
                    print(f"❌ API Error (לא quota): {str(e)[:100]}...")
                    return f"[API_ERROR] {str(e)}"

        # אם הגענו לכאן, כל הניסיונות נכשלו
        print(f"❌ כשל בכל {MAX_RETRIES} הניסיונות - מוותר על הטקסט")
        return "[MAX_RETRIES_EXCEEDED] לא הצליח לעבד אחרי מספר ניסיונות"

    def process_texts_parallel(self, texts):
        """עיבוד מקבילי של רשימת טקסטים עם טיפול משופר בשגיאות"""
        results = [''] * len(texts)  # שמירת סדר

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # שליחת בקשות עם index
            future_to_index = {
                executor.submit(self.clean_text_with_api, text): i
                for i, text in enumerate(texts)
            }

            # איסוף תוצאות
            completed_count = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                    completed_count += 1

                    # הדפסת התקדמות כל 10 טקסטים
                    if completed_count % 10 == 0:
                        print(f"    ✅ הושלמו {completed_count}/{len(texts)} טקסטים")

                except Exception as e:
                    results[index] = f"[THREAD_ERROR] {str(e)}"
                    print(f"❌ שגיאה בטקסט {index}: {str(e)[:50]}...")

        return results

    def process_single_file(self, file_key):
        """עיבוד קובץ יחיד"""
        file_name = os.path.basename(file_key)
        print(f"\n📁 מעבד: {file_name}")

        file_start_time = time.time()

        # קריאת קובץ
        df = self.read_csv_from_s3(SOURCE_BUCKET, file_key)
        if df is None:
            self.stats['errors'].append(f"לא הצליח לקרוא: {file_name}")
            return False

        print(f"📊 נטען: {len(df)} שורות")

        # בדיקת עמודת text
        if 'text' not in df.columns:
            print(f"❌ אין עמודת 'text' ב-{file_name}")
            self.stats['errors'].append(f"אין עמודת text: {file_name}")
            return False

        # הכנת טקסטים לעיבוד
        texts = df['text'].dropna().tolist()
        if not texts:
            print(f"⚠️ אין טקסטים ב-{file_name}")
            return False

        print(f"🔄 מעבד {len(texts)} טקסטים בבאצ'ים...")

        # עיבוד בבאצ'ים
        all_cleaned_texts = []
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in range(0, len(texts), BATCH_SIZE):
            batch_num = (batch_idx // BATCH_SIZE) + 1
            batch = texts[batch_idx:batch_idx + BATCH_SIZE]

            print(f"  📦 באצ' {batch_num}/{total_batches} ({len(batch)} טקסטים)")

            batch_start = time.time()
            cleaned_batch = self.process_texts_parallel(batch)
            batch_time = time.time() - batch_start

            all_cleaned_texts.extend(cleaned_batch)

            print(f"  ✅ הושלם: {batch_time:.1f}s ({batch_time / len(batch):.2f}s/text)")

            # השהיה בין באצ'ים למניעת rate limiting
            if batch_num < total_batches:
                sleep_time = 2 + random.uniform(0.5, 2.0)  # 2-4 שניות עם jitter
                print(f"  ⏳ המתנה {sleep_time:.1f}s לפני הבאצ' הבא...")
                time.sleep(sleep_time)

        # יצירת DataFrame עם תוצאות
        df_result = df.copy()
        df_result['cleaned_text'] = all_cleaned_texts[:len(df)]

        # שמירה ל-S3
        target_key = f"{TARGET_PREFIX}{PREFIX_FILTER.lower()}/{file_name}"
        success = self.save_csv_to_s3(df_result, TARGET_BUCKET, target_key)

        file_time = time.time() - file_start_time

        if success:
            print(f"✅ נשמר: s3://{TARGET_BUCKET}/{target_key}")
            print(f"⏰ זמן קובץ: {file_time / 60:.1f} דקות")

            # עדכון סטטיסטיקות
            self.stats['files_processed'] += 1
            self.stats['texts_processed'] += len(texts)
            self.stats['total_time'] += file_time

            return True
        else:
            self.stats['errors'].append(f"שגיאה בשמירת: {file_name}")
            return False

    def print_progress(self, current, total, start_time):
        """הדפסת התקדמות"""
        elapsed = time.time() - start_time
        if current > 0:
            avg_time_per_file = elapsed / current
            estimated_remaining = (total - current) * avg_time_per_file

            print(f"📊 התקדמות: {current}/{total} ({current / total * 100:.1f}%)")
            print(f"⏰ זמן שחלף: {elapsed / 60:.1f} דקות")
            print(f"🔮 זמן משוער לסיום: {estimated_remaining / 60:.1f} דקות")
            print(f"📈 קצב: {self.stats['texts_processed'] / elapsed:.1f} טקסטים/שנייה")

            # הוספת מידע על quota errors
            if self.stats['quota_errors'] > 0:
                print(f"⚠️ שגיאות Quota: {self.stats['quota_errors']} (ניסיונות חוזרים: {self.stats['retries']})")

    def print_final_stats(self):
        """הדפסת סטטיסטיקות סופיות"""
        total_time = time.time() - self.stats['start_time'].timestamp()

        print(f"\n{'=' * 60}")
        print("📊 סיכום עיבוד Geektime")
        print("=" * 60)
        print(f"✅ קבצים מעובדים: {self.stats['files_processed']}")
        print(f"📝 טקסטים מעובדים: {self.stats['texts_processed']:,}")
        print(f"⏰ זמן כולל: {total_time / 60:.1f} דקות")

        if self.stats['texts_processed'] > 0:
            avg_time = total_time / self.stats['texts_processed']
            print(f"⚡ זמן ממוצע לטקסט: {avg_time:.2f} שניות")
            print(f"🚀 קצב עיבוד: {self.stats['texts_processed'] / total_time:.1f} טקסטים/שנייה")

        # הצגת מידע על quota management
        if self.stats['quota_errors'] > 0:
            print(f"\n📊 ניהול Quota:")
            print(f"⚠️  שגיאות Quota: {self.stats['quota_errors']}")
            print(f"🔄 ניסיונות חוזרים: {self.stats['retries']}")
            success_rate = ((self.stats['texts_processed']) / (
                        self.stats['texts_processed'] + self.stats['quota_errors'])) * 100
            print(f"✅ אחוז הצלחה: {success_rate:.1f}%")

        if self.stats['errors']:
            print(f"\n❌ שגיאות ({len(self.stats['errors'])}):")
            for error in self.stats['errors'][:5]:  # רק 5 ראשונות
                print(f"  • {error}")
            if len(self.stats['errors']) > 5:
                print(f"  ... ועוד {len(self.stats['errors']) - 5} שגיאות")

        print(f"\n💾 קבצים נשמרו ב: s3://{TARGET_BUCKET}/{TARGET_PREFIX}{PREFIX_FILTER.lower()}/")

    def run_processing(self, test_mode=True):
        """הרצת עיבוד מלא"""
        print(f"🚀 מתחיל עיבוד קבצי {PREFIX_FILTER}")

        # מציאת קבצים
        limit = TEST_LIMIT if test_mode else None
        files = self.list_geektime_files(limit)

        if not files:
            print(f"❌ לא נמצאו קבצי {PREFIX_FILTER}")
            return

        print(f"\n🎯 מצב: {'בדיקה' if test_mode else 'ייצור'}")
        print(f"📁 מספר קבצים לעיבוד: {len(files)}")
        print(f"⚙️  הגדרות Quota Management:")
        print(f"   • Max Workers: {MAX_WORKERS}")
        print(f"   • Batch Size: {BATCH_SIZE}")
        print(f"   • Max Retries: {MAX_RETRIES}")
        print(f"   • Base Retry Delay: {BASE_RETRY_DELAY}s")

        # אישור משתמש
        if not test_mode:
            response = input(f"\n❓ להתחיל עיבוד {len(files)} קבצים? (y/N): ")
            if response.lower() != 'y':
                print("❌ עיבוד בוטל")
                return

        # עיבוד קבצים
        start_time = time.time()
        successful_files = 0

        for i, file_key in enumerate(files, 1):
            print(f"\n{'=' * 60}")
            print(f"📁 קובץ {i}/{len(files)}: {os.path.basename(file_key)}")

            success = self.process_single_file(file_key)
            if success:
                successful_files += 1

            # התקדמות כל 5 קבצים
            if i % 5 == 0 or i == len(files):
                self.print_progress(i, len(files), start_time)

        # סיכום
        self.print_final_stats()

        print(f"\n🎉 עיבוד הושלם!")
        print(f"✅ {successful_files}/{len(files)} קבצים עובדו בהצלחה")


def main():
    """פונקציה ראשית"""
    print("🚀 Gepeta Geektime Processor")
    print("=" * 60)

    try:
        # יצירת מעבד
        processor = GeektimeProcessor(GOOGLE_API_KEY)

        # בחירת מצב
        print(f"\n🎯 אפשרויות:")
        print(f"1. בדיקה ({TEST_LIMIT} קבצים)")
        print(f"2. ייצור (כל הקבצים)")

        choice = input("בחר (1/2): ").strip()

        if choice == "1":
            print(f"\n🧪 מתחיל בדיקה עם {TEST_LIMIT} קבצים...")
            processor.run_processing(test_mode=True)
        elif choice == "2":
            print(f"\n🏭 מתחיל עיבוד ייצור...")
            processor.run_processing(test_mode=False)
        else:
            print("❌ בחירה לא תקינה")
            return

    except ValueError as e:
        print(f"❌ {e}")
        print("💡 עדכן את GOOGLE_API_KEY בתחילת הקובץ או בקובץ .env")
    except Exception as e:
        print(f"❌ שגיאה: {e}")

    print(f"\n⏰ סיום: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()