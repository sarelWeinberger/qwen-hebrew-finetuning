#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import mwparserfromhell
import json
import re
import bz2
import csv
import ipaddress
import boto3
import shutil
from pathlib import Path
from tqdm import tqdm
import time
import os
from collections import defaultdict


class WikipediaProcessorRound2:
    def __init__(self, dump_path, s3_bucket, s3_prefix_main, s3_prefix_examples, max_articles=10000,
                 max_examples_per_category=100):
        self.dump_path = dump_path
        self.s3_bucket = s3_bucket
        self.s3_prefix_main = s3_prefix_main
        self.s3_prefix_examples = s3_prefix_examples
        self.max_articles = max_articles
        self.max_examples_per_category = max_examples_per_category

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

        # תיקיה מקומית זמנית ופתיחת קובץ הפלט הראשי
        self.temp_dir = Path("temp_output")
        self.temp_dir.mkdir(exist_ok=True)
        self.output_file = open(self.temp_dir / "wikipedia_he_round2.jsonl", 'w', encoding='utf-8')

    def upload_example_file_to_s3(self, category, temp_file):
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

    def upload_main_output_to_s3(self):
        """העלאת הקובץ הראשי ל-S3"""
        try:
            local_file = self.temp_dir / "wikipedia_he_round2.jsonl"
            s3_key = f"{self.s3_prefix_main}wikipedia_he_round2.jsonl"

            self.s3_client.upload_file(
                str(local_file),
                self.s3_bucket,
                s3_key
            )

            print(f"✅ הועלה קובץ ראשי: s3://{self.s3_bucket}/{s3_key}")
        except Exception as e:
            print(f"❌ שגיאה בהעלאת קובץ ראשי: {e}")

    def cleanup_temp_files(self):
        """ניקוי קבצים זמניים"""
        try:
            shutil.rmtree(self.temp_dir)
            print("🗑️ נוקו קבצים זמניים")
        except Exception as e:
            print(f"⚠️ שגיאה בניקוי קבצים זמניים: {e}")

    def save_example(self, category, raw_text, clean_text):
        """שמירת דוגמה לקטגוריה מסוימת ב-S3"""
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
        if self.example_counts[category] % 10 == 0 or self.example_counts[category] == self.max_examples_per_category:
            self.upload_example_file_to_s3(category, temp_file)

    def clean_html_escape_codes(self, text):
        """כלל 1: החלפת HTML escape codes"""
        original_text = text

        # החלפת escape codes שונים
        text = text.replace('&quot;', '"')
        text = text.replace('&#34;', '"')
        text = text.replace('&#39;', "'")

        # שמירת דוגמה אם השתנה משהו
        if text != original_text:
            self.save_example("html_escape_codes", original_text, text)

        return text

    def clean_newlines_and_spaces(self, text):
        """כלל 2: טיפול בשורות חדשות ורווחים"""
        original_text = text

        # הסרת carriage return
        text = text.replace('\r', '')

        # החלפת יותר מ-3 שורות חדשות רצופות במקסימום 3
        text = re.sub(r'\n{4,}', '\n\n\n', text)

        # שמירת דוגמה אם השתנה משהו
        if text != original_text:
            self.save_example("cr_and_3_newlines", original_text, text)

        return text

    def clean_multiple_spaces(self, text):
        """כלל 3: טיפול ברווחים מרובים"""
        original_text = text

        def replace_spaces(match):
            space_count = len(match.group(0))

            # אם מתחלק ב-4, להשאיר עד מקסימום 16
            if space_count % 4 == 0:
                return ' ' * min(space_count, 16)
            else:
                # אם לא מתחלק ב-4, לצמצם לרווח אחד
                return ' '

        # מציאת רצפים של 2+ רווחים
        text = re.sub(r' {2,}', replace_spaces, text)

        # שמירת דוגמה אם השתנה משהו
        if text != original_text:
            self.save_example("multiple_spaces", original_text, text)

        return text

    def clean_whitespace_start_end(self, text):
        """כלל 4: הסרת רווחים מתחילת וסוף הטקסט"""
        original_text = text
        text = text.strip()

        # שמירת דוגמה אם השתנה משהו
        if text != original_text:
            self.save_example("start_end_white_space", original_text, text)

        return text

    def is_localhost_ip(self, ip_str):
        """בדיקה אם IP הוא localhost (127.0.0.0/8)"""
        try:
            ip = ipaddress.ip_address(ip_str)
            localhost_network = ipaddress.ip_network('127.0.0.0/8')
            return ip in localhost_network
        except:
            return False

    def clean_pii(self, text):
        """כלל 6: מחיקת PII - IP ומייל (שמירת localhost)"""
        original_text = text

        # מחיקת כתובות מייל
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = re.sub(email_pattern, '[EMAIL_REMOVED]', text)

        # מחיקת IP addresses (חוץ מ-localhost)
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'

        def replace_ip(match):
            ip = match.group(0)
            if self.is_localhost_ip(ip):
                return ip  # שמירת localhost
            else:
                return '[IP_REMOVED]'

        text = re.sub(ip_pattern, replace_ip, text)

        # שמירת דוגמה אם השתנה משהו
        if text != original_text:
            self.save_example("PII", original_text, text)

        return text

    def clean_empty_bullet_lines(self, text):
        """כלל 7: הסרת שורות ריקות עם bullet points"""
        original_text = text

        # הסרת שורות שמכילות רק bullet points ורווחים
        bullet_pattern = r'^\s*[•●■◦▪◆]+\s*$'
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            if not re.match(bullet_pattern, line):
                cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)

        # שמירת דוגמה אם השתנה משהו
        if text != original_text:
            self.save_example("empty_line_with_bullet", original_text, text)

        return text

    def clean_separator_lines(self, text):
        """כלל 8: הסרת קווי הפרדה ארוכים"""
        original_text = text

        # הסרת שורות עם הפרדות ארוכות - רק תווי הפרדה באורך 4+
        separator_patterns = [
            r'^[-]{4,}$',  # מקפים בלבד
            r'^[=]{4,}$',  # סימני שווה בלבד
            r'^[_]{4,}$',  # קו תחתון בלבד
            r'^[~]{4,}$',  # טילדה בלבד
            r'^[*]{4,}$',  # כוכביות בלבד
            r'^[#]{4,}$',  # פאונד בלבד
        ]

        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line_stripped = line.strip()
            should_remove = False

            # בדיקה אם השורה מכילה רק תווי הפרדה (4 או יותר)
            for pattern in separator_patterns:
                if re.match(pattern, line_stripped):
                    should_remove = True
                    break

            # רק אם השורה לא צריכה להיות מוסרת, נכלול אותה
            if not should_remove:
                cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)

        # שמירת דוגמה אם השתנה משהו
        if text != original_text:
            self.save_example("multiple_hyphens", original_text, text)

        return text

    def clean_css_from_tables(self, text):
        """כלל 9: הסרת CSS מטבלאות"""
        original_text = text

        # הסרת style attributes
        text = re.sub(r'style="[^"]*"', '', text)
        text = re.sub(r'cellspacing="[^"]*"', '', text)
        text = re.sub(r'cellpadding="[^"]*"', '', text)
        text = re.sub(r'class="[^"]*"', '', text)
        text = re.sub(r'width="[^"]*"', '', text)
        text = re.sub(r'height="[^"]*"', '', text)

        # שמירת דוגמה אם השתנה משהו
        if text != original_text:
            self.save_example("CSS", original_text, text)

        return text

    def convert_tables_to_markdown(self, text):
        """כלל 10: המרת טבלאות ויקי למרקדאון"""
        original_text = text

        def process_wiki_table(match):
            table_content = match.group(0)
            table_inner = table_content[2:-2]  # הסר {| ו |}

            rows = re.split(r'\|-', table_inner)
            markdown_rows = []

            for i, row in enumerate(rows):
                row = row.strip()
                if not row:
                    continue

                # פיצול תאים
                cells = re.split(r'\|\||\|', row)
                clean_cells = []

                for cell in cells:
                    cell = cell.strip()
                    if cell and not re.match(r'^\d+px$', cell):
                        # ניקוי תא
                        cell = re.sub(r'^[\|\s]+', '', cell)
                        cell = re.sub(r'[\|\s]+$', '', cell)
                        if cell and len(cell) > 1:
                            clean_cells.append(cell)

                if clean_cells:
                    # יצירת שורת מרקדאון
                    markdown_row = "| " + " | ".join(clean_cells) + " |"
                    markdown_rows.append(markdown_row)

                    # הוספת שורת הפרדה אחרי השורה הראשונה
                    if i == 0 and len(clean_cells) > 0:
                        separator = "|" + "---|" * len(clean_cells)
                        markdown_rows.append(separator)

            return "\n".join(markdown_rows) if markdown_rows else ""

        # המרת טבלאות ויקי
        table_pattern = r'\{\|.*?\|\}'
        text = re.sub(table_pattern, process_wiki_table, text, flags=re.DOTALL)

        # שמירת דוגמה אם השתנה משהו
        if text != original_text:
            self.save_example("tables", original_text, text)

        return text

    def clean_wiki_templates_and_markup(self, text):
        """כלל 11a,b: הסרת תבניות ו-markup של ויקי"""
        original_text = text

        # הסרת תבניות מורכבות עם תבניות מקוננות {{}}
        # עושה זאת מספר פעמים כדי להתמודד עם קינון עמוק
        for _ in range(5):  # מקסימום 5 איטרציות
            old_text = text
            text = re.sub(r'\{\{[^{}]*\}\}', '', text)
            if text == old_text:  # אם לא השתנה, מפסיק
                break

        # הסרת קישורי קבצים/תמונות לפני קישורים רגילים
        media_link_patterns = [
            r'\[\[קובץ:.*?\]\]',
            r'\[\[File:.*?\]\]',
            r'\[\[Image:.*?\]\]',
            r'\[\[תמונה:.*?\]\]'
        ]

        for pattern in media_link_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

        # הסרת קישורים פנימיים [[]] עם תיאורים
        # מטפל בקישורים מהצורה [[קישור|תיאור]] → תיאור
        text = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', text)

        # הסרת קישורים פשוטים [[קישור]] → קישור
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)

        # הסרת תגיות HTML (כולל תגיות עם תכונות)
        text = re.sub(r'<[^>]*>', '', text)

        # ניקוי רווחים מיותרים ופסיקים שנותרו
        # הסרת פסיקים מיותרים שנותרו אחרי הסרת קישורים
        text = re.sub(r',\s*,', ',', text)  # פסיקים כפולים
        text = re.sub(r',\s*\]', ']', text)  # פסיק לפני סגירת סוגריים
        text = re.sub(r'\[\s*,', '[', text)  # פסיק אחרי פתיחת סוגריים

        # הסרת רווחים מרובים (רק רווחים וטאבים, לא שורות חדשות)
        text = re.sub(r'[ \t]+', ' ', text)

        # הסרת פסיקים בתחילת או סוף משפטים
        text = re.sub(r'^\s*,\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*,\s*$', '', text, flags=re.MULTILINE)

        # שמירת דוגמה אם השתנה משהו
        if text != original_text:
            self.save_example("wiki_markup", original_text, text)

        return text

    def clean_wiki_media_descriptions(self, text):
        """כלל 11b: הסרת תיאורי מדיה ותמונות"""
        original_text = text

        # הסרת תיאורי מיקום תמונות - באופן ספציפי
        location_keywords = ['שמאל', 'ימין', 'מרכז', 'ממוזער', 'thumb', 'thumbnail', 'frame', 'framed', 'left', 'right',
                             'center']

        for keyword in location_keywords:
            # הסרה ספציפית של תיאורי מיקום (כמו ממוזער|300px|)
            pattern = rf'\b{keyword}\|[^|]*?\|'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # הסרת הפניות "ראו" - תמיכה בעברית ואנגלית
        see_patterns = [
            r'ראו [A-Za-z\s,א-ת\.]+',
            r'ראה [A-Za-z\s,א-ת\.]+',
            r'see [A-Za-z\s,\.]+',
            r'See [A-Za-z\s,\.]+',
            r'ראו גם [A-Za-z\s,א-ת\.]+',
            r'ראה גם [A-Za-z\s,א-ת\.]+',
            r'see also [A-Za-z\s,\.]+',
            r'See also [A-Za-z\s,\.]+',
        ]

        for pattern in see_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # הסרת תיאורים בסוגריים שמכילים מידע על מדיה
        text = re.sub(r'\([^)]*(?:px|ממוזער|thumb|frame)[^)]*\)', '', text, flags=re.IGNORECASE)

        # הסרת מידע על גודל תמונות
        text = re.sub(r'\b\d+px\b', '', text)

        # ניקוי שורות ריקות עודפות שנוצרו
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)

        # ניקוי רווחים מיותרים (רק רווחים וטאבים, לא שורות חדשות)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)

        # שמירת דוגמה אם השתנה משהו
        if text != original_text:
            self.save_example("wiki_tags_and_media_descriptions", original_text, text)

        return text

    def clean_wiki_headers(self, text):
        """כלל 11d: המרת כותרות ויקי לפורמט מרקדאון"""
        original_text = text

        # המרת כותרות ויקי (== כותרת ==) למרקדאון (## כותרת)
        def convert_header(match):
            # מספר ה-= קובע את רמת הכותרת
            equals_prefix = match.group(1)
            header_text = match.group(2).strip()
            equals_suffix = match.group(3)

            # ספירת מספר ה-= (לוקח את המינימום בין התחלה וסוף)
            level = min(len(equals_prefix), len(equals_suffix))

            # המרה ל-markdown headers (מקסימום 6 רמות)
            markdown_level = min(level, 6)
            return '#' * markdown_level + ' ' + header_text

        # דפוס רגולרי משופר לזיהוי כותרות ויקי
        # מחפש: (=+) + רווחים אופציונלים + תוכן + רווחים אופציונלים + (=+)
        header_pattern = r'^(={2,6})\s*([^=\r\n]+?)\s*(={2,6})\s*$'

        text = re.sub(header_pattern, convert_header, text, flags=re.MULTILINE)

        # שמירת דוגמה אם השתנה משהו
        if text != original_text:
            self.save_example("wiki_headers", original_text, text)

        return text

    def clean_wiki_citations(self, text):
        """כלל 11e: הסרת citations"""
        original_text = text

        # הסרת citations שונים
        citation_patterns = [
            r'<ref[^>]*>.*?</ref>',  # ref tags
            r'<ref[^>]*/>',  # self-closing ref tags
            r'\{\{cite[^}]*\}\}',  # cite templates
            r'\{\{צ-[^}]*\}\}',  # Hebrew citations
            r'\{\{הערה[^}]*\}\}',  # Hebrew notes
            r'\{\{מקור[^}]*\}\}',  # Hebrew sources
        ]

        for pattern in citation_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

        # שמירת דוגמה אם השתנה משהו
        if text != original_text:
            self.save_example("wiki_citations", original_text, text)

        return text

    def check_redirecting_article(self, raw_wikitext):
        """כלל 11f: בדיקה אם המאמר הוא הפניה"""
        is_redirect = raw_wikitext.strip().startswith('#REDIRECT') or raw_wikitext.strip().startswith('#הפניה')

        if is_redirect:
            self.save_example("wiki_redirecting_articles", raw_wikitext, "[REDIRECTING ARTICLE - EXCLUDED]")

        return is_redirect

    def apply_all_cleaning_rules(self, title, raw_wikitext):
        """הפעלת כל כללי הניקיון על טקסט"""
        # בדיקה אם זה מאמר מפנה (כלל 11f)
        if self.check_redirecting_article(raw_wikitext):
            return None

        # המרה לאובייקט wikicode
        try:
            wikicode = mwparserfromhell.parse(raw_wikitext)
            text = str(wikicode)
        except:
            text = raw_wikitext

        # הפעלת כל הכללים
        text = self.clean_html_escape_codes(text)  # כלל 1
        text = self.clean_newlines_and_spaces(text)  # כלל 2
        text = self.clean_multiple_spaces(text)  # כלל 3
        text = self.clean_whitespace_start_end(text)  # כלל 4
        text = self.clean_pii(text)  # כלל 6
        text = self.clean_empty_bullet_lines(text)  # כלל 7
        text = self.clean_separator_lines(text)  # כלל 8
        text = self.clean_css_from_tables(text)  # כלל 9
        text = self.convert_tables_to_markdown(text)  # כלל 10
        text = self.clean_wiki_templates_and_markup(text)  # כלל 11a,b
        text = self.clean_wiki_media_descriptions(text)  # כלל 11b
        text = self.clean_wiki_headers(text)  # כלל 11d
        text = self.clean_wiki_citations(text)  # כלל 11e

        # בדיקה סופית של איכות
        if not text or len(text) < 100:
            return None

        return text

    def count_words(self, text):
        """ספירת מילים"""
        if not text:
            return 0
        return len(text.split())

    def count_bytes(self, text):
        """ספירת בייטים ב-UTF-8"""
        if not text:
            return 0
        return len(text.encode('utf-8'))

    def is_valid_article(self, page_elem):
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
        print("🚀 מתחיל עיבוד ויקיפדיה - סיבוב 2")
        print(f"📂 קובץ קלט: {self.dump_path}")
        print(f"☁️ S3 ראשי: s3://{self.s3_bucket}/{self.s3_prefix_main}")
        print(f"📁 S3 דוגמאות: s3://{self.s3_bucket}/{self.s3_prefix_examples}")
        print(f"🔢 מגבלה: {self.max_articles} ערכים")
        print(f"📊 דוגמאות: עד {self.max_examples_per_category} לכל קטגוריה")
        print("=" * 60)

        start_time = time.time()

        try:
            with bz2.open(self.dump_path, 'rt', encoding='utf-8') as dump_file:
                with tqdm(total=self.max_articles, desc="🔄 עיבוד ערכים", unit="articles") as pbar:

                    for event, elem in ET.iterparse(dump_file, events=('start', 'end')):
                        if event == 'end' and elem.tag.endswith('page'):
                            self.total_scanned += 1

                            # בדיקה אם הדף תקין
                            if self.is_valid_article(elem):
                                # חילוץ מידע
                                title_elem = elem.find('.//{*}title')
                                revision = elem.find('.//{*}revision')
                                text_elem = revision.find('.//{*}text')

                                title = title_elem.text if title_elem is not None else ""
                                raw_wikitext = text_elem.text if text_elem is not None else ""

                                # עיבוד הערך
                                cleaned_text = self.apply_all_cleaning_rules(title, raw_wikitext)

                                if cleaned_text:
                                    # יצירת פריט JSONL
                                    article_item = {
                                        "text": cleaned_text,
                                        "word_count": self.count_words(cleaned_text),
                                        "byte_count": self.count_bytes(cleaned_text)
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

            # העלאה ל-S3 של הקובץ הראשי
            self.upload_main_output_to_s3()

            # העלאה סופית של כל קבצי הדוגמאות שנותרו
            for category in self.example_categories:
                temp_file = self.temp_dir / f"{category}_temp.csv"
                if temp_file.exists() and self.example_counts[category] > 0:
                    self.upload_example_file_to_s3(category, temp_file)

            # ניקוי קבצים זמניים
            self.cleanup_temp_files()

        # סיכום סופי
        self.print_final_summary(start_time)

    def print_final_summary(self, start_time):
        """הדפסת סיכום סופי"""
        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("🎉 עיבוד הושלם!")
        print(f"📊 סטטיסטיקות סופיות:")
        print(f"   ⏱️ זמן כולל: {total_time / 60:.1f} דקות")
        print(f"   📖 דפים שנסרקו: {self.total_scanned:,}")
        print(f"   ✅ ערכים שעובדו: {self.total_processed:,}")
        print(f"   📈 שיעור הצלחה: {(self.total_processed / self.total_scanned) * 100:.2f}%")

        print(f"\n📊 דוגמאות שנשמרו:")
        for category, count in self.example_counts.items():
            if count > 0:
                print(f"   📁 {category}: {count} דוגמאות")

        print(f"\n☁️ הפלט זמין ב-S3:")
        print(f"   📄 קובץ ראשי: s3://{self.s3_bucket}/{self.s3_prefix_main}")
        print(f"   📁 דוגמאות: s3://{self.s3_bucket}/{self.s3_prefix_examples}")


def main():
    """הפעלה ראשית"""
    print("🎯 עיבוד ויקיפדיה עברית - סיבוב 2 עם כללי ניקיון מעודכנים")
    print("=" * 60)

    # הגדרות
    dump_path = r'C:\Users\Dan Revital\OneDrive\Documents\gepeta\data\wikipedia\hewiki-20250520-pages-articles.xml.bz2'
    s3_bucket = 'gepeta-datasets'
    s3_prefix_main = 'partly-processed/wikipedia_round_2/'
    s3_prefix_examples = 'partly-processed/round_2_test_examples/'
    max_articles = 1000  # הגבלה לבדיקה
    max_examples_per_category = 100

    # יצירת מעבד
    processor = WikipediaProcessorRound2(
        dump_path=dump_path,
        s3_bucket=s3_bucket,
        s3_prefix_main=s3_prefix_main,
        s3_prefix_examples=s3_prefix_examples,
        max_articles=max_articles,
        max_examples_per_category=max_examples_per_category
    )

    # בדיקת קישורית S3
    try:
        processor.s3_client.head_bucket(Bucket=s3_bucket)
        print(f"✅ קישורית S3 תקינה: s3://{s3_bucket}")
    except Exception as e:
        print(f"❌ בעיה בקישורית S3: {e}")
        return

    # הפעלת העיבוד
    print(f"\n🚀 מתחיל עיבוד עם כללי ניקיון מעודכנים...")
    processor.process_dump()


if __name__ == "__main__":
    main()