#!/usr/bin/env python3
"""
No-Truncation Test Tool - בדיקת הסרת חיתוך על 10 טקסטים ראשונים

בודק את הגרסה החדשה ללא חיתוך ה-[:800] על 10 טקסטים ראשונים
מדפיס עבור כל טקסט:
1. הטקסט המקורי + מספר מילים
2. הטקסט הנקי + מספר מילים
3. השוואה לגרסה הישנה עם חיתוך
"""

import google.generativeai as genai
import boto3
import pandas as pd
from io import StringIO
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# =============================================================================
# הגדרות
# =============================================================================

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY_SANDBOX_2", "YOUR_API_KEY_HERE")

# הגדרות קובץ לבדיקה
TEST_BUCKET = "gepeta-datasets"
TEST_PREFIX = "partly-processed/regex-and-dedup"
TEST_FILE_PATTERN = "Geektime"  # נקח את הקובץ הראשון


class NoTruncationTestTool:
    """כלי לבדיקת הסרת חיתוך"""

    def __init__(self, api_key):
        if api_key == "YOUR_API_KEY_HERE":
            raise ValueError("❌ עדכן את GOOGLE_API_KEY!")

        # Google AI Setup
        genai.configure(api_key=api_key)
        self.model_name = 'gemini-2.0-flash'

        # S3 Setup
        self.s3_client = boto3.client('s3')

    def count_words(self, text):
        """ספירת מילים בטקסט"""
        if pd.isna(text) or text == '':
            return 0
        words = str(text).split()
        return len(words)

    def clean_text_old_version(self, text):
        """ניקוי טקסט עם חיתוך ישן ([:800])"""
        model = genai.GenerativeModel(self.model_name)

        prompt = f"""נקה את הטקסט העברי הבא מפגמי קידוד, תגיות HTML, פרסומות ותבניות. החזר רק טקסט נקי בעברית:

{text[:800]}

טקסט נקי:"""

        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"[API_ERROR] {str(e)}"

    def clean_text_new_version(self, text):
        """ניקוי טקסט חדש - ללא חיתוך"""
        model = genai.GenerativeModel(self.model_name)

        prompt = f"""נקה את הטקסט העברי הבא מפגמי קידוד, תגיות HTML, פרסומות ותבניות. החזר רק טקסט נקי בעברית:

{text}

טקסט נקי:"""

        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"[API_ERROR] {str(e)}"

    def find_first_geektime_file(self):
        """מציאת הקובץ הראשון של Geektime"""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')

            for page in paginator.paginate(Bucket=TEST_BUCKET, Prefix=TEST_PREFIX):
                if 'Contents' not in page:
                    continue

                for obj in page['Contents']:
                    key = obj['Key']
                    filename = os.path.basename(key)

                    if (key.endswith('.csv') and
                            TEST_FILE_PATTERN in filename and
                            obj['Size'] > 0):
                        return key

        except Exception as e:
            print(f"❌ שגיאה בחיפוש קבצים: {e}")

        return None

    def read_csv_from_s3(self, bucket, key):
        """קריאת CSV מ-S3"""
        try:
            print(f"📖 קורא: {os.path.basename(key)}")
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(content))
            return df
        except Exception as e:
            print(f"❌ שגיאה בקריאת {key}: {e}")
            return None

    def run_test(self):
        """הרצת בדיקה על 10 טקסטים ראשונים"""
        print("🧪 No-Truncation Test Tool")
        print("=" * 80)
        print(f"🤖 מודל: {self.model_name}")
        print(f"🔍 בדיקה: גרסה ישנה ([:800]) VS גרסה חדשה (טקסט מלא)")
        print()

        # מציאת הקובץ הראשון
        file_key = self.find_first_geektime_file()
        if not file_key:
            print("❌ לא נמצא קובץ Geektime")
            return

        # קריאת הקובץ
        df = self.read_csv_from_s3(TEST_BUCKET, file_key)
        if df is None:
            return

        if 'text' not in df.columns:
            print("❌ אין עמודת 'text' בקובץ")
            return

        # בדיקת 10 הטקסטים הראשונים
        texts_to_test = df['text'].dropna().head(10).tolist()

        print(f"📊 בודק {len(texts_to_test)} טקסטים ראשונים מ-{os.path.basename(file_key)}")
        print("=" * 80)

        total_start_time = time.time()

        for i, original_text in enumerate(texts_to_test, 1):
            print(f"\n📝 טקסט #{i}")
            print("-" * 60)

            # ספירת מילים במקור
            original_word_count = self.count_words(original_text)
            original_char_count = len(original_text)

            print(f"📊 מקור: {original_word_count} מילים, {original_char_count} תווים")

            # הדפסת הטקסט המקורי (מקוצר למסך)
            print(f"📄 טקסט מקורי:")
            print(f"   {original_text[:150]}...")
            print()

            # עיבוד עם הגרסה הישנה (חיתוך)
            print(f"🔄 מעבד עם חיתוך ישן ([:800])...")
            old_start = time.time()
            old_cleaned_text = self.clean_text_old_version(original_text)
            old_time = time.time() - old_start
            old_word_count = self.count_words(old_cleaned_text)

            print(f"⏰ זמן גרסה ישנה: {old_time:.2f}s")
            print(f"✨ תוצאה ישנה: {old_word_count} מילים")
            print(f"📄 טקסט ישן:")
            print(f"   {old_cleaned_text[:150]}...")
            print()

            # המתנה קצרה בין הבקשות
            time.sleep(1)

            # עיבוד עם הגרסה החדשה (ללא חיתוך)
            print(f"🔄 מעבד ללא חיתוך (טקסט מלא)...")
            new_start = time.time()
            new_cleaned_text = self.clean_text_new_version(original_text)
            new_time = time.time() - new_start
            new_word_count = self.count_words(new_cleaned_text)

            print(f"⏰ זמן גרסה חדשה: {new_time:.2f}s")
            print(f"✨ תוצאה חדשה: {new_word_count} מילים")
            print(f"📄 טקסט חדש:")
            print(f"   {new_cleaned_text[:150]}...")

            # השוואה
            print(f"\n📈 השוואה:")
            if original_word_count > 0:
                old_retention = (old_word_count / original_word_count) * 100
                new_retention = (new_word_count / original_word_count) * 100
                print(f"   גרסה ישנה: {old_retention:.1f}% שימור מילים")
                print(f"   גרסה חדשה: {new_retention:.1f}% שימור מילים")
                improvement = new_retention - old_retention
                print(f"   שיפור: {improvement:+.1f}% מילים")

            time_diff = new_time - old_time
            print(f"   הבדל זמן: {time_diff:+.2f}s ({'+איטי יותר' if time_diff > 0 else 'מהיר יותר'})")

            # בדיקה אם הטקסט נחתך
            if original_char_count > 800:
                print(f"   📏 הטקסט המקורי היה {original_char_count} תווים (מעל 800)")
                if new_word_count > old_word_count * 1.2:  # שיפור של לפחות 20%
                    print(f"   ✅ הגרסה החדשה שיפרה משמעותית את השימור!")
                else:
                    print(f"   ⚠️ השיפור לא משמעותי")
            else:
                print(f"   📏 הטקסט המקורי היה {original_char_count} תווים (מתחת ל-800)")
                print(f"   💡 לא אמור להיות הבדל גדול")

            print("-" * 60)

        total_time = time.time() - total_start_time

        print(f"\n🎉 בדיקה הושלמה!")
        print(f"⏰ זמן כולל: {total_time / 60:.2f} דקות")
        print(f"📊 ממוצע זמן לטקסט: {total_time / len(texts_to_test):.2f} שניות")
        print(f"🔧 בדיקה זו עזרה לקבוע אם הסרת החיתוך פותרת את בעיית החיתוך")


def main():
    """פונקציה ראשית"""
    try:
        tool = NoTruncationTestTool(GOOGLE_API_KEY)
        tool.run_test()

    except ValueError as e:
        print(f"❌ {e}")
        print("💡 עדכן את GOOGLE_API_KEY בתחילת הקובץ או בקובץ .env")
    except Exception as e:
        print(f"❌ שגיאה כללית: {e}")


if __name__ == "__main__":
    main()