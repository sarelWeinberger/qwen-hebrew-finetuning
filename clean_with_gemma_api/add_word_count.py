#!/usr/bin/env python3
"""
Dataset Summary Tool - מחשב סיכום מילים עבור דטאסטים מעובדים

קורא קבצים מעובדים מ-S3 ומחשב:
- text_words (סיכום n_count מינוס 1)
- clean_text_words (ספירת מילים ב-cleaned_text)

יוצר קובץ CSV עם סיכום לכל דטאסט
"""

import boto3
import pandas as pd
from io import StringIO
import os
from datetime import datetime

# =============================================================================
# הגדרות דטאסטים
# =============================================================================

DATASETS_CONFIG = [
    {
        'name': 'Geektime',
        'type': 'folder',  # תיקיה שלמה
        'bucket': 'gepeta-datasets',
        'path': 'processed/geektime/'
    },
    {
        'name': 'YisraelHayom',
        'type': 'file',  # קובץ יחיד
        'bucket': 'gepeta-datasets',
        'path': 'processed/yisraelhayom/YisraelHayomData-Combined-Deduped.forgpt_part-0_cleaned.csv'
    }
]

# הגדרות פלט
OUTPUT_BUCKET = 'gepeta-datasets'
OUTPUT_PREFIX = 'summaries/'


class DatasetSummaryTool:
    """כלי לסיכום סטטיסטיקות דטאסטים מעובדים"""

    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.summary_data = []

    def count_words(self, text):
        """ספירת מילים בטקסט"""
        if pd.isna(text) or text == '':
            return 0
        words = str(text).split()
        return len(words)

    def read_csv_from_s3(self, bucket, key):
        """קריאת CSV מ-S3"""
        try:
            print(f"  📖 קורא: {os.path.basename(key)}")
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(content))
            return df
        except Exception as e:
            print(f"  ❌ שגיאה בקריאת {key}: {e}")
            return None

    def list_files_in_folder(self, bucket, prefix):
        """רשימת קבצים בתיקיה"""
        files = []
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')

            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if 'Contents' not in page:
                    continue

                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.csv') and obj['Size'] > 0:
                        files.append(key)

        except Exception as e:
            print(f"❌ שגיאה ברישום קבצים מ-{prefix}: {e}")

        return files

    def process_dataset(self, dataset_config):
        """עיבוד דטאסט יחיד"""
        dataset_name = dataset_config['name']
        dataset_type = dataset_config['type']
        bucket = dataset_config['bucket']
        path = dataset_config['path']

        print(f"\n🔍 מעבד דטאסט: {dataset_name}")
        print(f"📁 מיקום: s3://{bucket}/{path}")

        if dataset_type == 'folder':
            # תיקיה שלמה
            files = self.list_files_in_folder(bucket, path)
            if not files:
                print(f"⚠️ לא נמצאו קבצים ב-{path}")
                return

            print(f"✅ נמצאו {len(files)} קבצים")

            # עיבוד כל הקבצים בתיקיה
            total_text_words = 0
            total_clean_text_words = 0
            processed_files = 0

            for file_key in files:
                df = self.read_csv_from_s3(bucket, file_key)
                if df is None:
                    continue

                # בדיקת עמודות נדרשות
                if 'n_count' not in df.columns or 'cleaned_text' not in df.columns:
                    print(f"  ⚠️ חסרות עמודות ב-{os.path.basename(file_key)}")
                    continue

                # חישוב מילים במקור (מינוס 1 מהסך הכל - תיקון הכותרת)
                file_text_words = df['n_count'].sum() - 1

                # חישוב מילים בטקסט נקי
                file_clean_words = df['cleaned_text'].apply(self.count_words).sum()

                total_text_words += file_text_words
                total_clean_text_words += file_clean_words
                processed_files += 1

                print(f"  ✅ {os.path.basename(file_key)}: {file_text_words:,} → {file_clean_words:,} מילים")

        elif dataset_type == 'file':
            # קובץ יחיד
            df = self.read_csv_from_s3(bucket, path)
            if df is None:
                return

            # בדיקת עמודות נדרשות
            if 'n_count' not in df.columns or 'cleaned_text' not in df.columns:
                print(f"❌ חסרות עמודות בקובץ")
                return

            # חישוב מילים במקור (מינוס 1 מהסך הכל - תיקון הכותרת)
            total_text_words = df['n_count'].sum() - 1

            # חישוב מילים בטקסט נקי
            total_clean_text_words = df['cleaned_text'].apply(self.count_words).sum()
            processed_files = 1

            print(f"  ✅ {os.path.basename(path)}: {total_text_words:,} → {total_clean_text_words:,} מילים")

        else:
            print(f"❌ סוג דטאסט לא מוכר: {dataset_type}")
            return

        # הדפסת סיכום הדטאסט
        print(f"\n📊 סיכום {dataset_name}:")
        print(f"  📄 קבצים מעובדים: {processed_files}")
        print(f"  📝 מילים מקוריות: {total_text_words:,}")
        print(f"  ✨ מילים נקיות: {total_clean_text_words:,}")
        if total_text_words > 0:
            reduction = ((total_text_words - total_clean_text_words) / total_text_words * 100)
            print(f"  📉 הפחתה: {reduction:.1f}%")

        # הוספה לסיכום הכללי
        self.summary_data.append({
            'Dataset': dataset_name,
            'text_words': total_text_words,
            'clean_text_words': total_clean_text_words
        })

    def save_summary_to_s3(self):
        """שמירת הסיכום ל-S3"""
        try:
            # יצירת DataFrame
            summary_df = pd.DataFrame(self.summary_data)

            # הוספת שורת סיכום כללי
            if len(summary_df) > 1:
                total_row = {
                    'Dataset': 'TOTAL',
                    'text_words': summary_df['text_words'].sum(),
                    'clean_text_words': summary_df['clean_text_words'].sum()
                }
                summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)

            # שמירה ל-S3
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_key = f"{OUTPUT_PREFIX}dataset_word_summary_{timestamp}.csv"

            csv_buffer = StringIO()
            summary_df.to_csv(csv_buffer, index=False, encoding='utf-8')

            self.s3_client.put_object(
                Bucket=OUTPUT_BUCKET,
                Key=output_key,
                Body=csv_buffer.getvalue(),
                ContentType='text/csv'
            )

            print(f"\n✅ סיכום נשמר ב: s3://{OUTPUT_BUCKET}/{output_key}")

            # הדפסת הסיכום
            print(f"\n📋 סיכום סופי:")
            print(summary_df.to_string(index=False))

            return True

        except Exception as e:
            print(f"❌ שגיאה בשמירת הסיכום: {e}")
            return False

    def run_summary(self):
        """הרצת סיכום מלא"""
        print("🚀 מתחיל סיכום דטאסטים מעובדים")
        print("=" * 60)

        # עיבוד כל הדטאסטים
        for dataset_config in DATASETS_CONFIG:
            self.process_dataset(dataset_config)

        # שמירת הסיכום
        success = self.save_summary_to_s3()

        if success:
            print(f"\n🎉 סיכום הושלם בהצלחה!")
        else:
            print(f"\n❌ שגיאה בשמירת הסיכום")


def main():
    """פונקציה ראשית"""
    print("🔍 Dataset Summary Tool")
    print("=" * 60)
    print("📊 מחשב סיכום מילים עבור דטאסטים מעובדים")
    print()

    print("📋 דטאסטים שיעובדו:")
    for i, dataset in enumerate(DATASETS_CONFIG, 1):
        print(f"  {i}. {dataset['name']}: s3://{dataset['bucket']}/{dataset['path']}")

    print()

    try:
        tool = DatasetSummaryTool()
        tool.run_summary()

    except Exception as e:
        print(f"❌ שגיאה כללית: {e}")

    print(f"\n⏰ סיום: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()