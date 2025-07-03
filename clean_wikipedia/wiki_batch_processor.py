#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikipedia Batch Processor
==========================

עיבוד מלא של דאמפ ויקיפדיה עברית עם שמירת דוגמאות ל-S3.
משתמש במודול הניקוי המרכזי.
"""

import xml.etree.ElementTree as ET
import json
import bz2
import csv
import boto3
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import time
from collections import defaultdict

from wiki_text_cleaner import WikipediaTextCleaner, count_words, count_bytes

WIKI_ARTICLES = 500000 # Estimated total no. of articles, for the progress bar, in case we prcoess them all

class WikipediaBatchProcessor:
    """מחלקה לעיבוד מלא של דאמפ ויקיפדיה עם שמירת דוגמאות"""

    def __init__(self, dump_path, s3_bucket, s3_prefix_main, s3_prefix_examples,
                 max_articles=10000, max_examples_per_category=100, max_random_samples=100):
        self.dump_path = dump_path
        self.s3_bucket = s3_bucket
        self.s3_prefix_main = s3_prefix_main
        self.s3_prefix_examples = s3_prefix_examples
        self.max_articles = max_articles
        self.max_examples_per_category = max_examples_per_category
        self.max_random_samples = max_random_samples

        # S3 client
        self.s3_client = boto3.client('s3')

        # תיקיות דוגמאות (קיימות ב-S3)
        self.example_categories = [
            "cr_and_3_newlines", "CSS", "empty_line_with_bullet", "equations",
            "html_escape_codes", "multiple_hyphens", "multiple_spaces", "PII",
            "start_end_white_space", "tables", "URL", "wiki_citations",
            "wiki_foreign_language_and_image_refs", "wiki_headers", "wiki_markup",
            "wiki_redirecting_articles", "wiki_tags_and_media_descriptions"
        ]

        # מונים לדוגמאות
        self.example_counts = defaultdict(int)

        # מונים כלליים
        self.total_processed = 0
        self.total_scanned = 0

        # דגימה אקראית כללית
        self.random_sample_indices = set()
        self.random_samples_collected = 0
        self.random_samples_data = []

        # יצירת רשימה של אינדקסים אקראיים לדגימה
        if self.max_random_samples > 0:
            # בחירת אינדקסים אקראיים מתוך הריצה
            self.random_sample_indices = set(random.sample(
                range(min(self.max_articles, 50000)),
                min(self.max_random_samples, self.max_articles)
            ))

        # תיקיה מקומית זמנית ופתיחת קובץ הפלט הראשי
        self.temp_dir = Path("temp_output")
        self.temp_dir.mkdir(exist_ok=True)
        self.output_file = open(self.temp_dir / "wikipedia_he_processed.jsonl", 'w', encoding='utf-8')

        # יצירת מנקה עם callback לשמירת דוגמאות
        self.cleaner = WikipediaTextCleaner(example_callback=self._save_example)

    def _save_example(self, category, raw_text, clean_text):
        """callback לשמירת דוגמאות לקטגוריה מסוימת"""
        if self.example_counts[category] >= self.max_examples_per_category:
            return

        # יצירת קובץ CSV זמני
        temp_file = self.temp_dir / f"{category}_temp.csv"

        # בדיקה אם הקובץ קיים ויצירת headers
        file_exists = temp_file.exists()

        with open(temp_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(['raw_text', 'clean_text'])

            writer.writerow([raw_text, clean_text])

        self.example_counts[category] += 1

        # העלאה ל-S3 כל 10 דוגמאות או בסוף
        if (self.example_counts[category] % 10 == 0 or
                self.example_counts[category] == self.max_examples_per_category):
            self._upload_example_file_to_s3(category, temp_file)

    def _save_random_sample(self, raw_wikitext, clean_text, title):
        """שמירת דגימה אקראית כללית"""
        if self.random_samples_collected >= self.max_random_samples:
            return

        # הוספת הדגימה לרשימה
        self.random_samples_data.append({
            'raw_text': raw_wikitext,
            'clean_text': clean_text,
            'title': title
        })

        self.random_samples_collected += 1

        # שמירה לקובץ זמני כל 10 דגימות או בסוף
        if (self.random_samples_collected % 10 == 0 or
                self.random_samples_collected == self.max_random_samples):
            self._write_random_samples_to_file()

    def _write_random_samples_to_file(self):
        """כתיבת הדגימות האקראיות לקובץ זמני"""
        temp_file = self.temp_dir / "random_samples_temp.csv"

        # בדיקה אם הקובץ קיים ויצירת headers
        file_exists = temp_file.exists()

        with open(temp_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(['raw_text', 'clean_text', 'title'])

            # כתיבת כל הדגימות שנאספו
            for sample in self.random_samples_data:
                writer.writerow([sample['raw_text'], sample['clean_text'], sample['title']])

        # ניקוי הרשימה אחרי כתיבה
        self.random_samples_data.clear()

    def _upload_example_file_to_s3(self, category, temp_file):
        """העלאת קובץ דוגמאות ל-S3"""
        try:
            s3_key = f"{self.s3_prefix_examples}{category}/{category}_examples.csv"

            self.s3_client.upload_file(
                str(temp_file),
                self.s3_bucket,
                s3_key
            )

            print(f"✅ הועלו דוגמאות {category}: {self.example_counts[category]} דוגמאות")
        except Exception as e:
            print(f"❌ שגיאה בהעלאת דוגמאות {category}: {e}")

    def _upload_random_samples_to_s3(self):
        """העלאת הדגימות האקראיות ל-S3"""
        try:
            temp_file = self.temp_dir / "random_samples_temp.csv"
            if not temp_file.exists():
                return

            s3_key = f"{self.s3_prefix_examples}random_samples/random_samples.csv"

            self.s3_client.upload_file(
                str(temp_file),
                self.s3_bucket,
                s3_key
            )

            print(f"✅ הועלו דגימות אקראיות כלליות: {self.random_samples_collected} דגימות")
        except Exception as e:
            print(f"❌ שגיאה בהעלאת דגימות אקראיות: {e}")

    def _upload_main_output_to_s3(self):
        """העלאת הקובץ הראשי ל-S3"""
        try:
            local_file = self.temp_dir / "wikipedia_he_processed.jsonl"
            s3_key = f"{self.s3_prefix_main}wikipedia_he_processed.jsonl"

            self.s3_client.upload_file(
                str(local_file),
                self.s3_bucket,
                s3_key
            )

            print(f"✅ הועלה קובץ ראשי: s3://{self.s3_bucket}/{s3_key}")
        except Exception as e:
            print(f"❌ שגיאה בהעלאת קובץ ראשי: {e}")

    def _cleanup_temp_files(self):
        """ניקוי קבצים זמניים"""
        try:
            shutil.rmtree(self.temp_dir)
            print("🗑️ נוקו קבצים זמניים")
        except Exception as e:
            print(f"⚠️ שגיאה בניקוי קבצים זמניים: {e}")

    def _is_valid_article(self, page_elem):
        """בדיקה אם הדף תקין לעיבוד"""
        namespace_elem = page_elem.find('.//{*}ns')
        if namespace_elem is None or namespace_elem.text != '0':
            return False

        revision = page_elem.find('.//{*}revision')
        if revision is None:
            return False

        text_elem = revision.find('.//{*}text')
        if text_elem is None or not text_elem.text:
            return False

        return True

    def process_dump(self):
        """עיבוד הדאמפ עם הגבלה"""
        print("🚀 מתחיל עיבוד ויקיפדיה - עם מנקה מאוחד")
        print(f"📂 קובץ קלט: {self.dump_path}")
        print(f"☁️ S3 ראשי: s3://{self.s3_bucket}/{self.s3_prefix_main}")
        print(f"📁 S3 דוגמאות: s3://{self.s3_bucket}/{self.s3_prefix_examples}")
        print(f"🔢 מגבלה: {self.max_articles} ערכים")
        print(f"📊 דוגמאות: עד {self.max_examples_per_category} לכל קטגוריה")
        print(f"🎲 דגימות אקראיות כלליות: {self.max_random_samples}")
        print("=" * 60)

        start_time = time.time()

        try:
            with bz2.open(self.dump_path, 'rt', encoding='utf-8') as dump_file:
                with tqdm(total=self.max_articles, desc="🔄 עיבוד ערכים", unit="articles") as pbar:

                    for event, elem in ET.iterparse(dump_file, events=('start', 'end')):
                        if event == 'end' and elem.tag.endswith('page'):
                            self.total_scanned += 1

                            # בדיקה אם הדף תקין
                            if self._is_valid_article(elem):
                                # חילוץ מידע
                                title_elem = elem.find('.//{*}title')
                                revision = elem.find('.//{*}revision')
                                text_elem = revision.find('.//{*}text')

                                title = title_elem.text if title_elem is not None else ""
                                raw_wikitext = text_elem.text if text_elem is not None else ""

                                # עיבוד הערך באמצעות המנקה המרכזי
                                cleaned_text = self.cleaner.clean_article(title, raw_wikitext)

                                if cleaned_text:
                                    # בדיקה אם צריך לשמור דגימה אקראית
                                    if (self.total_processed in self.random_sample_indices and
                                            self.random_samples_collected < self.max_random_samples):
                                        self._save_random_sample(raw_wikitext, cleaned_text, title)

                                    # יצירת פריט JSONL
                                    article_item = {
                                        "text": cleaned_text,
                                        "word_count": count_words(cleaned_text),
                                        "byte_count": count_bytes(cleaned_text),
                                        "title": title
                                    }

                                    # כתיבה לקובץ הפלט
                                    json.dump(article_item, self.output_file, ensure_ascii=False)
                                    self.output_file.write('\n')

                                    self.total_processed += 1
                                    pbar.update(1)
                                    pbar.set_description(f"🔄 עיבוד: {title[:25]}...")

                                    # בדיקה אם הגענו למגבלה
                                    if self.total_processed >= self.max_articles:
                                        break

                            elem.clear()

                        # בדיקה אם הגענו למגבלה
                        if self.total_processed >= self.max_articles:
                            break

        except KeyboardInterrupt:
            print("\n⚠️ העיבוד הופסק על ידי המשתמש")
        except Exception as e:
            print(f"\n❌ שגיאה בעיבוד: {e}")

        finally:
            self.output_file.close()

            # וידוא שמירת כל הדגימות האקראיות שנותרו
            if self.random_samples_data:
                self._write_random_samples_to_file()

            # העלאה ל-S3 של הקובץ הראשי
            self._upload_main_output_to_s3()

            # העלאה של הדגימות האקראיות
            self._upload_random_samples_to_s3()

            # העלאה סופית של כל קבצי הדוגמאות שנותרו
            for category in self.example_categories:
                temp_file = self.temp_dir / f"{category}_temp.csv"
                if temp_file.exists() and self.example_counts[category] > 0:
                    self._upload_example_file_to_s3(category, temp_file)

            # ניקוי קבצים זמניים
            self._cleanup_temp_files()

        # סיכום סופי
        self._print_final_summary(start_time)

    def _print_final_summary(self, start_time):
        """הדפסת סיכום סופי"""
        total_time = time.time() - start_time
        cleaning_stats = self.cleaner.get_stats()

        print("\n" + "=" * 60)
        print("🎉 עיבוד הושלם!")
        print(f"📊 סטטיסטיקות סופיות:")
        print(f"   ⏱️ זמן כולל: {total_time / 60:.1f} דקות")
        print(f"   📖 דפים שנסרקו: {self.total_scanned:,}")
        print(f"   ✅ ערכים שעובדו: {self.total_processed:,}")
        print(f"   📈 שיעור הצלחה: {(self.total_processed / self.total_scanned) * 100:.2f}%")

        print(f"\n🔧 סטטיסטיקות ניקוי:")
        for stat_name, count in cleaning_stats.items():
            if count > 0:
                print(f"   📊 {stat_name}: {count:,}")

        print(f"\n📊 דוגמאות שנשמרו:")
        for category, count in self.example_counts.items():
            if count > 0:
                print(f"   📁 {category}: {count} דוגמאות")

        # הוספת סטטיסטיקה על הדגימות האקראיות
        print(f"\n🎲 דגימות אקראיות כלליות:")
        print(f"   📊 נאספו: {self.random_samples_collected} מתוך {self.max_random_samples}")

        print(f"\n☁️ הפלט זמין ב-S3:")
        print(f"   📄 קובץ ראשי: s3://{self.s3_bucket}/{self.s3_prefix_main}")
        print(f"   📁 דוגמאות: s3://{self.s3_bucket}/{self.s3_prefix_examples}")
        print(f"   🎲 דגימות אקראיות: s3://{self.s3_bucket}/{self.s3_prefix_examples}random_samples/")


def main():
    """הפעלה ראשית"""
    print("🎯 עיבוד ויקיפדיה עברית - מערכת מאוחדת")
    print("=" * 60)

    # הגדרות
    dump_path = r'C:\Users\Dan Revital\OneDrive\Documents\gepeta\data\wikipedia\hewiki-latest-pages-articles.xml.bz2'
    s3_bucket = 'gepeta-datasets'
    s3_prefix_main = 'processed_and_cleaned/wikipedia/'
    s3_prefix_examples = 'processed/unified_examples/'
    max_articles = WIKI_ARTICLES  # הגבלה אפשרית לבדיקה
    max_examples_per_category = 0
    max_random_samples = 0  # דגימות אקראיות כלליות

    # יצירת מעבד
    processor = WikipediaBatchProcessor(
        dump_path=dump_path,
        s3_bucket=s3_bucket,
        s3_prefix_main=s3_prefix_main,
        s3_prefix_examples=s3_prefix_examples,
        max_articles=max_articles,
        max_examples_per_category=max_examples_per_category,
        max_random_samples=max_random_samples
    )

    # בדיקת קישורית S3
    try:
        processor.s3_client.head_bucket(Bucket=s3_bucket)
        print(f"✅ קישורית S3 תקינה: s3://{s3_bucket}")
    except Exception as e:
        print(f"❌ בעיה בקישורית S3: {e}")
        return

    # הפעלת העיבוד
    print(f"\n🚀 מתחיל עיבוד עם מערכת ניקוי מאוחדת...")
    processor.process_dump()


if __name__ == "__main__":
    main()