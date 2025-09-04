#!/usr/bin/env python3
"""
Gepeta Project - Geektime Files Processor
עיבוד קבצי Geektime עם Google Gemini API + מעקב מילים

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
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import json

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
MAX_WORKERS = 10  # מספר threads מקבילים (הגדרה מיטבית שמצאנו)
BATCH_SIZE = 50  # גודל באצ' לעיבוד
TEST_LIMIT = 100  # מספר קבצים לבדיקה (None לכל הקבצים)


class GeektimeProcessor:
    """מעבד קבצי Geektime עם Google API"""

    def __init__(self, api_key):
        """אתחול המעבד"""
        if api_key == "YOUR_API_KEY_HERE":
            raise ValueError("❌ עדכן את GOOGLE_API_KEY!")

        # Google AI Setup
        genai.configure(api_key=api_key)
        self.model_name = 'gemini-2.0-flash'  # שדרוג למודל 2.0

        # S3 Setup
        self.s3_client = boto3.client('s3')

        # Statistics - הוספת מעקב מילים
        self.stats = {
            'files_processed': 0,
            'texts_processed': 0,
            'total_time': 0,
            'errors': [],
            'total_original_words': 0,  # חדש: סיכום מילים מקוריות
            'total_cleaned_words': 0,  # חדש: סיכום מילים נקיות
            'start_time': datetime.now()
        }

        print("🚀 Gepeta Generic Prefix Processor מוכן")
        print(f"🤖 מודל: {self.model_name}")
        print(f"📁 מקור: s3://{SOURCE_BUCKET}/{SOURCE_PREFIX}")
        print(f"🎯 יעד: s3://{TARGET_BUCKET}/{TARGET_PREFIX}{PREFIX_FILTER.lower()}/")
        print(f"🔍 קידומת: {PREFIX_FILTER}")
        print(f"👥 Workers: {MAX_WORKERS}")
        print(f"⏰ התחלה: {self.stats['start_time'].strftime('%H:%M:%S')}")

    def count_words(self, text):
        """ספירת מילים בטקסט"""
        if pd.isna(text) or text == '':
            return 0
        words = str(text).split()
        return len(words)

    def extract_part_number(self, filename):
        """חילוץ מספר part מהשם קובץ"""
        # דוגמה: GeektimeCorpus-Combined-Deduped.forgpt_part-1_cleaned.csv
        # מחפש תבנית: part-{number}
        match = re.search(r'part-(\d+)', filename)
        if match:
            return int(match.group(1))
        return None

    def list_geektime_files(self, limit=None, part_start=None, part_end=None):
        """מציאת כל קבצי Geektime עם סינון לפי part range"""
        print(f"🔍 מחפש קבצי {PREFIX_FILTER}...")

        if part_start is not None and part_end is not None:
            print(f"🎯 מסנן: part-{part_start} עד part-{part_end}")
        elif part_start is not None:
            print(f"🎯 מסנן: החל מ-part-{part_start}")
        elif part_end is not None:
            print(f"🎯 מסנן: עד part-{part_end}")

        files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')

        page_count = 0
        for page in paginator.paginate(Bucket=SOURCE_BUCKET, Prefix=SOURCE_PREFIX):
            page_count += 1
            print(f"  סורק דף {page_count}...")

            if 'Contents' not in page:
                continue

            page_files = []
            for obj in page['Contents']:
                key = obj['Key']
                filename = os.path.basename(key)

                # בדיקות בסיסיות
                if not (key.endswith('.csv') and
                        PREFIX_FILTER in filename and
                        obj['Size'] > 0):
                    continue

                # בדיקת part number אם יש סינון
                if part_start is not None or part_end is not None:
                    part_num = self.extract_part_number(filename)
                    if part_num is None:
                        continue  # דלג על קבצים בלי part number

                    # בדיקת טווח
                    if part_start is not None and part_num < part_start:
                        continue
                    if part_end is not None and part_num > part_end:
                        continue

                page_files.append(key)

            files.extend(page_files)
            print(f"  דף {page_count}: נמצאו {len(page_files)} קבצי {PREFIX_FILTER}")

            # הגבלה לבדיקה
            if limit and len(files) >= limit:
                files = files[:limit]
                break

        # מיון לפי part number
        def sort_key(file_path):
            filename = os.path.basename(file_path)
            part_num = self.extract_part_number(filename)
            return part_num if part_num is not None else -1

        files.sort(key=sort_key)

        print(f"✅ נמצאו {len(files)} קבצי {PREFIX_FILTER} כולל")

        # הדפסת מספר parts שנמצאו
        if files:
            parts = [self.extract_part_number(os.path.basename(f)) for f in files]
            parts = [p for p in parts if p is not None]
            if parts:
                print(f"📊 טווח parts: {min(parts)} - {max(parts)}")

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

    def clean_text_with_api(self, text):
        """ניקוי טקסט יחיד עם Google API - ללא חיתוך"""
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
        """עיבוד מקבילי של רשימת טקסטים"""
        results = [''] * len(texts)  # שמירת סדר

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # שליחת בקשות עם index
            future_to_index = {
                executor.submit(self.clean_text_with_api, text): i
                for i, text in enumerate(texts)
            }

            # איסוף תוצאות
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    results[index] = f"[ERROR] {str(e)}"

        return results

    def process_single_file(self, file_key):
        """עיבוד קובץ יחיד"""
        file_name = os.path.basename(file_key)
        part_num = self.extract_part_number(file_name)
        part_info = f" (part-{part_num})" if part_num is not None else ""

        print(f"\n📁 מעבד: {file_name}{part_info}")

        file_start_time = time.time()

        # קריאת קובץ
        df = self.read_csv_from_s3(SOURCE_BUCKET, file_key)
        if df is None:
            self.stats['errors'].append(f"לא הצליח לקרוא: {file_name}")
            return False

        print(f"📊 נטען: {len(df)} שורות")

        # בדיקת עמודות נדרשות
        if 'text' not in df.columns:
            print(f"❌ אין עמודת 'text' ב-{file_name}")
            self.stats['errors'].append(f"אין עמודת text: {file_name}")
            return False

        if 'n_count' not in df.columns:
            print(f"❌ אין עמודת 'n_count' ב-{file_name}")
            self.stats['errors'].append(f"אין עמודת n_count: {file_name}")
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

            # המתנה קטנה למניעת rate limiting
            if batch_num < total_batches:
                time.sleep(0.5)

        # יצירת DataFrame עם תוצאות
        df_result = df.copy()
        df_result['cleaned_text'] = all_cleaned_texts[:len(df)]

        # חישוב סטטיסטיקות מילים לקובץ הנוכחי
        file_original_words = (df_result['n_count'].sum() - 1)  # מינוס 1 מהסך הכל
        file_cleaned_words = df_result['cleaned_text'].apply(self.count_words).sum()

        print(f"📝 מילים מקוריות: {file_original_words:,}")
        print(f"✨ מילים נקיות: {file_cleaned_words:,}")
        if file_original_words > 0:
            reduction = ((file_original_words - file_cleaned_words) / file_original_words * 100)
            print(f"📉 הפחתה: {reduction:.1f}%")

        # שמירה ל-S3
        target_key = f"{TARGET_PREFIX}{PREFIX_FILTER.lower()}/{file_name}"
        success = self.save_csv_to_s3(df_result, TARGET_BUCKET, target_key)

        file_time = time.time() - file_start_time

        if success:
            print(f"✅ נשמר: s3://{TARGET_BUCKET}/{target_key}")
            print(f"⏰ זמן קובץ: {file_time / 60:.1f} דקות")

            # עדכון סטטיסטיקות כוללות
            self.stats['files_processed'] += 1
            self.stats['texts_processed'] += len(texts)
            self.stats['total_time'] += file_time
            self.stats['total_original_words'] += file_original_words  # חדש
            self.stats['total_cleaned_words'] += file_cleaned_words  # חדש

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

            # הוספת מידע על מילים
            if self.stats['total_original_words'] > 0:
                reduction = ((self.stats['total_original_words'] - self.stats['total_cleaned_words']) /
                             self.stats['total_original_words'] * 100)
                print(
                    f"📝 מילים עד כה: {self.stats['total_original_words']:,} → {self.stats['total_cleaned_words']:,} ({reduction:.1f}% הפחתה)")

    def save_dataset_summary(self):
        """שמירת סיכום מילים של הדטאסט הנוכחי"""
        if self.stats['total_original_words'] == 0:
            return

        try:
            # יצירת שורת סיכום
            summary_data = {
                'Dataset': PREFIX_FILTER,
                'text_words': self.stats['total_original_words'],
                'clean_text_words': self.stats['total_cleaned_words']
            }

            summary_df = pd.DataFrame([summary_data])

            # שמירה ל-S3
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_key = f"summaries/{PREFIX_FILTER.lower()}_word_summary_{timestamp}.csv"

            csv_buffer = StringIO()
            summary_df.to_csv(csv_buffer, index=False, encoding='utf-8')

            self.s3_client.put_object(
                Bucket=TARGET_BUCKET,
                Key=summary_key,
                Body=csv_buffer.getvalue(),
                ContentType='text/csv'
            )

            print(f"📊 סיכום מילים נשמר ב: s3://{TARGET_BUCKET}/{summary_key}")

        except Exception as e:
            print(f"⚠️ שגיאה בשמירת סיכום מילים: {e}")

    def print_final_stats(self):
        """הדפסת סטטיסטיקות סופיות"""
        total_time = time.time() - self.stats['start_time'].timestamp()

        print(f"\n{'=' * 60}")
        print("📊 סיכום עיבוד")
        print("=" * 60)
        print(f"✅ קבצים מעובדים: {self.stats['files_processed']}")
        print(f"📝 טקסטים מעובדים: {self.stats['texts_processed']:,}")
        print(f"⏰ זמן כולל: {total_time / 60:.1f} דקות")

        if self.stats['texts_processed'] > 0:
            avg_time = total_time / self.stats['texts_processed']
            print(f"⚡ זמן ממוצע לטקסט: {avg_time:.2f} שניות")
            print(f"🚀 קצב עיבוד: {self.stats['texts_processed'] / total_time:.1f} טקסטים/שנייה")

        # הצגת מידע על מילים
        if self.stats['total_original_words'] > 0:
            print(f"\n📝 סיכום מילים:")
            print(f"   מילים מקוריות: {self.stats['total_original_words']:,}")
            print(f"   מילים נקיות: {self.stats['total_cleaned_words']:,}")
            reduction = ((self.stats['total_original_words'] - self.stats['total_cleaned_words']) /
                         self.stats['total_original_words'] * 100)
            print(f"   הפחתה: {reduction:.1f}%")

        if self.stats['errors']:
            print(f"\n❌ שגיאות ({len(self.stats['errors'])}):")
            for error in self.stats['errors'][:5]:  # רק 5 ראשונות
                print(f"  • {error}")
            if len(self.stats['errors']) > 5:
                print(f"  ... ועוד {len(self.stats['errors']) - 5} שגיאות")

        print(f"\n💾 קבצים נשמרו ב: s3://{TARGET_BUCKET}/{TARGET_PREFIX}{PREFIX_FILTER.lower()}/")

        # שמירת סיכום מילים
        self.save_dataset_summary()

    def get_part_range_input(self):
        """קבלת טווח parts מהמשתמש"""
        print(f"\n🎯 הגדרת טווח Parts לעיבוד:")
        print(f"💡 פורמט: part-0, part-1, part-2, ...")
        print(f"📝 ברירת מחדל: כל ה-parts")

        try:
            start_input = input("🔢 Part התחלה (Enter לכל הטווח): ").strip()
            part_start = None if start_input == "" else int(start_input)

            end_input = input("🔢 Part סיום (Enter לכל הטווח): ").strip()
            part_end = None if end_input == "" else int(end_input)

            # ולידציה
            if part_start is not None and part_end is not None and part_start > part_end:
                print("❌ שגיאה: Part התחלה גדול מ-Part הסיום")
                return self.get_part_range_input()  # נסה שוב

            if part_start is not None and part_start < 0:
                print("❌ שגיאה: Part חייב להיות מספר חיובי")
                return self.get_part_range_input()  # נסה שוב

            if part_end is not None and part_end < 0:
                print("❌ שגיאה: Part חייב להיות מספר חיובי")
                return self.get_part_range_input()  # נסה שוב

            # הדפסת בחירה
            if part_start is not None and part_end is not None:
                print(f"✅ נבחר טווח: part-{part_start} עד part-{part_end}")
            elif part_start is not None:
                print(f"✅ נבחר: החל מ-part-{part_start}")
            elif part_end is not None:
                print(f"✅ נבחר: עד part-{part_end}")
            else:
                print(f"✅ נבחר: כל ה-parts")

            return part_start, part_end

        except ValueError:
            print("❌ שגיאה: יש להזין מספר שלם בלבד")
            return self.get_part_range_input()  # נסה שוב

    def run_processing(self, test_mode=True):
        """הרצת עיבוד מלא"""
        print(f"🚀 מתחיל עיבוד קבצי {PREFIX_FILTER}")

        # קבלת טווח parts
        part_start, part_end = self.get_part_range_input()

        # מציאת קבצים
        limit = TEST_LIMIT if test_mode else None
        files = self.list_geektime_files(limit, part_start, part_end)

        if not files:
            print(f"❌ לא נמצאו קבצי {PREFIX_FILTER} בטווח שנבחר")
            return

        print(f"\n🎯 מצב: {'בדיקה' if test_mode else 'ייצור'}")
        print(f"📁 מספר קבצים לעיבוד: {len(files)}")

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
    print("🚀 Gepeta Processor (No Truncation)")
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