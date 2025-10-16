#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import mwparserfromhell
import json
import re
import bz2
import boto3
from pathlib import Path
from tqdm import tqdm
import time
import os


class FullWikipediaProcessor:
    def __init__(self, dump_path, s3_bucket, s3_prefix, chunk_size=50000):
        self.dump_path = dump_path
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.chunk_size = chunk_size
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

        # S3 client
        self.s3_client = boto3.client('s3')

        # מונים
        self.total_processed = 0
        self.total_scanned = 0
        self.current_chunk = 1
        self.current_chunk_count = 0
        self.current_file = None
        self.uploaded_files = []

    def normalize_text_for_training(self, text):
        """נרמול טקסט (זהה לחלוטין לתוכנית המדגם)"""
        if not text or not isinstance(text, str):
            return text

        # תיקון תווי בריחה לפני הכל
        text = text.replace('\\"', '"')
        text = text.replace("\\'", "'")
        text = text.replace('\\\\', '\\')

        # תיקון גם וריאציות של תווי בריחה
        text = text.replace('&quot;', '"')
        text = text.replace('&#34;', '"')
        text = text.replace('&#39;', "'")

        # מחק שורות חדשות ו-\r
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')

        # מחק תווי בריחה שנשארו
        text = text.replace('\\n', ' ')
        text = text.replace('\\t', ' ')
        text = text.replace('\\r', ' ')

        # רווחים מרובים → רווח יחיד
        text = re.sub(r'\s+', ' ', text)

        # נקה רווחים בהתחלה/סוף
        text = text.strip()

        return text

    def convert_wiki_tables_to_lists(self, content):
        """המרת טבלאות ויקי לרשימות נקיות"""

        def process_single_table(match):
            table_content = match.group(0)
            table_inner = table_content[2:-2]  # הסר {| ו |}

            rows = re.split(r'\|-', table_inner)
            items = []

            for row in rows:
                row = row.strip()
                if not row:
                    continue

                # הסרת סגנונות CSS
                row = re.sub(r'style="[^"]*"', '', row)
                row = re.sub(r'cellspacing="[^"]*"', '', row)
                row = re.sub(r'cellpadding="[^"]*"', '', row)
                row = re.sub(r'\|\s*\d+px', '', row)

                # פיצול תאים
                cells = re.split(r'\|\||\|', row)

                for cell in cells:
                    cell = cell.strip()
                    if cell and not re.match(r'^\d+px$', cell):
                        cell = re.sub(r'^[\|\s]+', '', cell)
                        cell = re.sub(r'[\|\s]+$', '', cell)
                        if cell and len(cell) > 1:
                            items.append(cell)

            if items:
                if len(items) >= 3:
                    result = "## נושאים קשורים:\n"
                    for item in items:
                        result += "- " + item + "\n"
                    return result
                else:
                    return " ".join(items)

            return ""

        table_pattern = r'\{\|.*?\|\}'
        cleaned_content = re.sub(table_pattern, process_single_table, content, flags=re.DOTALL)
        return cleaned_content

    def apply_regex_cleaning(self, content):
        """ניקוי regex (עם תוספת טיפול בטבלאות)"""
        # המרת טבלאות ויקי לרשימות
        content = self.convert_wiki_tables_to_lists(content)

        # הסרת תבניות שנשארו
        content = re.sub(r'\{\{[^}]*\}\}', '', content)
        content = re.sub(r'\[\[[^]]*\]\]', '', content)
        content = re.sub(r'<[^>]*>', '', content)

        # הסרת תיאורי תמונות ומדיה
        content = re.sub(r'^(שמאל|ימין|מרכז|ממוזער)\|.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*(שמאל|ימין|מרכז|ממוזער)\|.*$', '', content, flags=re.MULTILINE)

        # הסרת הסברי שפות זרות
        foreign_languages = ['בגרמנית', 'בהונגרית', 'בערבית', 'בכורדית', 'באנגלית',
                             'בצרפתית', 'באיטלקית', 'ברוסית', 'ביוונית', 'בלטינית']
        for lang in foreign_languages:
            pattern = r'\(' + lang + r':.*?\)'
            content = re.sub(pattern, '', content)

        # הסרת הפניות לתמונות
        content = re.sub(r'ראו [A-Za-z\s,]+\.', '', content)

        # ניקוי כללי
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)
        content = re.sub(r'^\s*=+.*?=+\s*$', '', content, flags=re.MULTILINE)

        return content

    def identify_headers(self, content):
        """זיהוי וסימון כותרות"""
        lines = content.split('\n')
        processed_lines = []

        for i, line in enumerate(lines):
            line = line.strip()

            if (line and len(line) < 100 and
                    i < len(lines) - 1 and
                    len(lines[i + 1].strip()) > 50 and
                    line.count('.') <= 1 and
                    line.count(',') <= 2):

                header_keywords = ['היסטוריה', 'ביוגרפיה', 'רקע', 'תולדות', 'מוצא', 'תרבות', 'משפחתו', 'ילדותו',
                                   'נעוריו', 'התפתחות', 'משימות']
                if (not line.endswith('.') or
                        any(keyword in line for keyword in header_keywords)):
                    processed_lines.append("## " + line)
                else:
                    processed_lines.append(line)
            else:
                processed_lines.append(line)

        return '\n'.join(processed_lines)

    def clean_content(self, wikicode):
        """ניקוי תוכן (זהה לחלוטין לתוכנית המדגם)"""
        try:
            # הסרת תבניות
            templates_to_remove = []
            for template in wikicode.filter_templates():
                template_name = str(template.name).strip().lower()
                remove_patterns = [
                    'cite', 'צ-', 'הערה', 'מקור', 'reflist', 'מקורות',
                    'ציון', 'ref', 'citation', 'web', 'news', 'book', 'journal'
                ]
                if any(pattern in template_name for pattern in remove_patterns):
                    templates_to_remove.append(template)

            for template in templates_to_remove:
                try:
                    wikicode.remove(template)
                except:
                    pass

            # המרת קישורים פנימיים לטקסט
            for link in wikicode.filter_wikilinks():
                try:
                    if link.text:
                        wikicode.replace(link, str(link.text))
                    else:
                        title = str(link.title)
                        if '|' in title:
                            title = title.split('|')[0]
                        wikicode.replace(link, title)
                except:
                    pass

            # הסרת תגיות
            for tag in wikicode.filter_tags():
                try:
                    if tag.tag.lower() in ['math', 'chem']:
                        wikicode.replace(tag, "[נוסחה: " + str(tag.contents) + "]")
                    elif tag.tag.lower() in ['ref', 'references']:
                        wikicode.remove(tag)
                except:
                    pass

            # המרת הכל לטקסט
            content = str(wikicode.strip_code())

            # ניקוי regex
            content = self.apply_regex_cleaning(content)

            # זיהוי כותרות
            content = self.identify_headers(content)

            # נרמול הטקסט לאימון
            content = self.normalize_text_for_training(content)

            return content.strip()

        except Exception as e:
            print("Error in clean_content: " + str(e))
            basic_clean = str(wikicode)[:2000]
            return self.normalize_text_for_training(basic_clean)

    def count_words(self, text):
        """ספירת מילים"""
        if not text:
            return 0
        words = text.split()
        return len(words)

    def count_bytes(self, text):
        """ספירת בייטים ב-UTF-8"""
        if not text:
            return 0
        return len(text.encode('utf-8'))

    def process_article(self, title, raw_wikitext):
        """עיבוד ערך יחיד ליצירת פריט JSONL"""
        try:
            wikicode = mwparserfromhell.parse(raw_wikitext)
            processed_text = self.clean_content(wikicode)

            if not processed_text or len(processed_text) < 100:
                return None

            article_item = {
                "text": processed_text,
                "word_count": self.count_words(processed_text),
                "byte_count": self.count_bytes(processed_text)
            }

            return article_item

        except Exception as e:
            return None

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

        if len(text_elem.text) < 500:
            return False

        if text_elem.text.strip().startswith('#REDIRECT'):
            return False

        return True

    def get_current_filename(self):
        """יוצר שם קובץ לchunk הנוכחי"""
        return self.output_dir / f"wikipedia_he_part_{self.current_chunk:03d}.jsonl"

    def open_new_chunk_file(self):
        """פותח קובץ חדש לchunk"""
        if self.current_file:
            self.current_file.close()

        filename = self.get_current_filename()
        self.current_file = open(filename, 'w', encoding='utf-8')
        self.current_chunk_count = 0

        print(f"\n📂 פותח chunk חדש: {filename}")

    def write_article_to_current_chunk(self, article_item):
        """כותב ערך לchunk הנוכחי"""
        if self.current_file is None or self.current_chunk_count >= self.chunk_size:
            if self.current_file:
                self.current_file.close()
                # העלאה של הchunk שהסתיים
                self.upload_chunk_to_s3(self.get_current_filename())
                self.current_chunk += 1

            self.open_new_chunk_file()

        json.dump(article_item, self.current_file, ensure_ascii=False)
        self.current_file.write('\n')
        self.current_chunk_count += 1

    def upload_chunk_to_s3(self, local_file):
        """מעלה chunk ל-S3"""
        s3_key = f"{self.s3_prefix}{local_file.name}"

        try:
            print(f"⬆️ מעלה ל-S3: {s3_key}")

            # בדיקת גודל קובץ
            file_size_mb = local_file.stat().st_size / (1024 * 1024)

            start_time = time.time()
            self.s3_client.upload_file(
                str(local_file),
                self.s3_bucket,
                s3_key
            )
            upload_time = time.time() - start_time

            # אימות שההעלאה הצליחה
            try:
                self.s3_client.head_object(Bucket=self.s3_bucket, Key=s3_key)
                print(f"✅ הועלה בהצלחה: {file_size_mb:.1f}MB ב-{upload_time:.1f}s")
                self.uploaded_files.append(s3_key)

                # מחיקת הקובץ המקומי לחסכון במקום
                local_file.unlink()
                print(f"🗑️ נמחק קובץ מקומי: {local_file.name}")

            except:
                print(f"❌ שגיאה באימות העלאה: {s3_key}")

        except Exception as e:
            print(f"❌ שגיאה בהעלאה: {e}")

    def estimate_total_articles(self):
        """אומדן מספר הערכים הכולל בדאמפ"""
        print("📊 מעריך את מספר הערכים הכולל...")

        sample_size = 10000  # דגימה של 10K דפים
        valid_count = 0
        total_scanned = 0

        try:
            with bz2.open(self.dump_path, 'rt', encoding='utf-8') as dump_file:
                for event, elem in ET.iterparse(dump_file, events=('start', 'end')):
                    if event == 'end' and elem.tag.endswith('page'):
                        total_scanned += 1

                        if self.is_valid_article(elem):
                            valid_count += 1

                        elem.clear()

                        if total_scanned >= sample_size:
                            break

            # חישוב אומדן על בסיס הדגימה
            if total_scanned > 0:
                success_rate = valid_count / total_scanned

                # אומדן גס של סך הדפים בויקיפדיה העברית
                estimated_total_pages = 1500000  # אומדן בסיס
                estimated_valid_articles = int(estimated_total_pages * success_rate)

                print(f"✅ דגימה של {total_scanned:,} דפים:")
                print(f"   📈 שיעור הצלחה: {success_rate * 100:.1f}%")
                print(f"   🎯 אומדן ערכים תקינים: {estimated_valid_articles:,}")

                return estimated_valid_articles

        except Exception as e:
            print(f"⚠️ לא הצלחתי להעריך את המספר הכולל: {e}")

        # fallback אם האומדן נכשל
        return 800000  # אומדן ברירת מחדל

    def process_full_dump(self):
        """עיבוד הדאמפ המלא"""
        print("🚀 מתחיל עיבוד מלא של ויקיפדיה")
        print("📂 דאמפ: " + self.dump_path)
        print("☁️ S3: s3://" + self.s3_bucket + "/" + self.s3_prefix)
        print("📦 גודל chunk: " + str(self.chunk_size) + " ערכים")
        print("=" * 60)

        # אומדן מספר הערכים הכולל
        estimated_total = self.estimate_total_articles()
        print("=" * 60)

        start_time = time.time()

        # פתיחת הchunk הראשון
        self.open_new_chunk_file()

        try:
            with bz2.open(self.dump_path, 'rt', encoding='utf-8') as dump_file:

                # יצירת progress bar עם מספר כולל משוער
                with tqdm(total=estimated_total, desc="🔄 עיבוד ערכים", unit="articles") as pbar:

                    for event, elem in ET.iterparse(dump_file, events=('start', 'end')):
                        if event == 'end' and elem.tag.endswith('page'):
                            self.total_scanned += 1

                            # עדכון התקדמות כל 500 דפים (יותר תכוף לעדכון זמן מדויק יותר)
                            if self.total_scanned % 500 == 0:
                                # חישוב הערכת זמן
                                elapsed_time = time.time() - start_time
                                if self.total_processed > 0:
                                    articles_per_second = self.total_processed / elapsed_time
                                    remaining_articles = estimated_total - self.total_processed
                                    eta_seconds = remaining_articles / articles_per_second if articles_per_second > 0 else 0
                                    eta_hours = eta_seconds / 3600

                                    pbar.set_postfix_str(
                                        f"סרקתי: {self.total_scanned:,} | "
                                        f"chunk: {self.current_chunk} ({self.current_chunk_count:,}/{self.chunk_size:,}) | "
                                        f"ETA: {eta_hours:.1f}h | "
                                        f"קצב: {articles_per_second:.1f} ערכים/שנ"
                                    )
                                else:
                                    pbar.set_postfix_str(
                                        f"סרקתי: {self.total_scanned:,} | "
                                        f"chunk: {self.current_chunk} ({self.current_chunk_count:,}/{self.chunk_size:,})"
                                    )

                            # בדיקה אם הדף תקין
                            if self.is_valid_article(elem):

                                # חילוץ מידע
                                title_elem = elem.find('.//{*}title')
                                revision = elem.find('.//{*}revision')
                                text_elem = revision.find('.//{*}text')

                                title = title_elem.text if title_elem is not None else ""
                                raw_wikitext = text_elem.text if text_elem is not None else ""

                                # עיבוד הערך
                                article_item = self.process_article(title, raw_wikitext)

                                if article_item:
                                    # כתיבה לchunk הנוכחי
                                    self.write_article_to_current_chunk(article_item)

                                    self.total_processed += 1
                                    pbar.update(1)

                                    # עדכון התיאור הנוכחי
                                    pbar.set_description(f"🔄 עיבוד: {title[:25]}...")

                            elem.clear()

            # סגירת הchunk האחרון
            if self.current_file:
                self.current_file.close()
                if self.current_chunk_count > 0:  # רק אם יש תוכן
                    self.upload_chunk_to_s3(self.get_current_filename())

        except KeyboardInterrupt:
            print("\n⚠️ העיבוד הופסק על ידי המשתמש")
            if self.current_file:
                self.current_file.close()
                if self.current_chunk_count > 0:
                    self.upload_chunk_to_s3(self.get_current_filename())

        except Exception as e:
            print(f"\n❌ שגיאה בעיבוד: {e}")
            if self.current_file:
                self.current_file.close()

        # סיכום סופי
        self.print_final_summary(start_time)

    def print_final_summary(self, start_time):
        """הדפסת סיכום סופי"""
        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("🎉 עיבוד הושלם!")
        print("📊 סטטיסטיקות סופיות:")
        print(f"   ⏱️ זמן כולל: {total_time / 3600:.1f} שעות")
        print(f"   📖 דפים שנסרקו: {self.total_scanned:,}")
        print(f"   ✅ ערכים שעובדו: {self.total_processed:,}")
        print(f"   📈 שיעור הצלחה: {(self.total_processed / self.total_scanned) * 100:.2f}%")
        print(f"   📦 chunks שנוצרו: {len(self.uploaded_files)}")
        print(f"   ⚡ מהירות עיבוד: {self.total_processed / (total_time / 3600):.0f} ערכים/שעה")

        print(f"\n☁️ קבצים ב-S3:")
        for i, s3_key in enumerate(self.uploaded_files, 1):
            print(f"   {i:2d}. s3://{self.s3_bucket}/{s3_key}")

        print(f"\n✅ הדאטאסט זמין ב: s3://{self.s3_bucket}/{self.s3_prefix}")


def main():
    """הפעלה ראשית"""
    print("🎯 עיבוד מלא של ויקיפדיה ל-JSONL + העלאה ל-S3")
    print("=" * 60)

    # הגדרות
    dump_path = r'C:\Users\Dan Revital\OneDrive\Documents\gepeta\data\wikipedia\hewiki-latest-pages-articles.xml.bz2'
    s3_bucket = 'gepeta-datasets'
    s3_prefix = 'processed/wikipedia/'
    chunk_size = 50000

    # יצירת מעבד
    processor = FullWikipediaProcessor(
        dump_path=dump_path,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        chunk_size=chunk_size
    )

    # בדיקת קישורית S3
    try:
        processor.s3_client.head_bucket(Bucket=s3_bucket)
        print(f"✅ קישורית S3 תקינה: s3://{s3_bucket}")
    except Exception as e:
        print(f"❌ בעיה בקישורית S3: {e}")
        return

    # הפעלת העיבוד
    print(f"\n🚀 מתחיל עיבוד...")
    processor.process_full_dump()


if __name__ == "__main__":
    main()