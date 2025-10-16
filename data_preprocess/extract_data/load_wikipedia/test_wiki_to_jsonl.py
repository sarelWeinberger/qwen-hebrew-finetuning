#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import mwparserfromhell
import json
import re
import bz2
from pathlib import Path
from tqdm import tqdm


class WikipediaJSONLProcessor:
    def __init__(self, dump_path, output_file, max_articles=100):
        self.dump_path = dump_path
        self.output_file = output_file
        self.max_articles = max_articles
        self.processed_articles = []

    def normalize_text_for_training(self, text):
        """
        נרמול טקסט (זהה לחלוטין לתוכנית המדגם)
        """
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
        """
        המרת טבלאות ויקי לרשימות נקיות
        """

        def process_single_table(match):
            table_content = match.group(0)

            # הסרת תגית הפתיחה והסגירה
            table_inner = table_content[2:-2]  # הסר {| ו |}

            # פיצול לשורות
            rows = re.split(r'\|-', table_inner)

            items = []

            for row in rows:
                row = row.strip()
                if not row:
                    continue

                # הסרת סגנונות CSS מהשורה הראשונה
                row = re.sub(r'style="[^"]*"', '', row)
                row = re.sub(r'cellspacing="[^"]*"', '', row)
                row = re.sub(r'cellpadding="[^"]*"', '', row)
                row = re.sub(r'\|\s*\d+px', '', row)  # הסר מידות כמו 96px

                # פיצול תאים - יכול להיות || או |
                cells = re.split(r'\|\||\|', row)

                for cell in cells:
                    cell = cell.strip()
                    if cell and not re.match(r'^\d+px$', cell):  # לא תא של מידות
                        # נקה רווחים וסימנים מיותרים
                        cell = re.sub(r'^[\|\s]+', '', cell)
                        cell = re.sub(r'[\|\s]+$', '', cell)
                        if cell and len(cell) > 1:  # רק תאים משמעותיים
                            items.append(cell)

            # אם נמצאו פריטים, המר לרשימה
            if items:
                # נסה לזהות אם יש כותרת הגיונית
                if len(items) >= 3:
                    result = "## נושאים קשורים:\n"
                    for item in items:
                        result += "- " + item + "\n"
                    return result
                else:
                    # אם יש מעט פריטים, רק רשימה פשוטה
                    return " ".join(items)

            return ""  # אם לא נמצא תוכן משמעותי

        # מציאה והחלפה של כל הטבלאות
        table_pattern = r'\{\|.*?\|\}'
        cleaned_content = re.sub(table_pattern, process_single_table, content, flags=re.DOTALL)

        return cleaned_content

    def apply_regex_cleaning(self, content):
        """
        ניקוי regex (עם תוספת טיפול בטבלאות)
        """
        # המרת טבלאות ויקי לרשימות - לפני ניקויים אחרים
        content = self.convert_wiki_tables_to_lists(content)

        # הסרת תבניות שנשארו
        content = re.sub(r'\{\{[^}]*\}\}', '', content)
        # הסרת קישורים שנשארו
        content = re.sub(r'\[\[[^]]*\]\]', '', content)
        # הסרת תגיות HTML
        content = re.sub(r'<[^>]*>', '', content)

        # הסרת תיאורי תמונות ומדיה
        content = re.sub(r'^(שמאל|ימין|מרכז|ממוזער)\|.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*(שמאל|ימין|מרכז|ממוזער)\|.*$', '', content, flags=re.MULTILINE)

        # הסרת הסברי שפות זרות בסוגריים
        foreign_languages = ['בגרמנית', 'בהונגרית', 'בערבית', 'בכורדית', 'באנגלית',
                             'בצרפתית', 'באיטלקית', 'ברוסית', 'ביוונית', 'בלטינית']
        for lang in foreign_languages:
            pattern = r'\(' + lang + r':.*?\)'
            content = re.sub(pattern, '', content)

        # הסרת הפניות לתמונות
        content = re.sub(r'ראו [A-Za-z\s,]+\.', '', content)

        # ניקוי כללי
        content = re.sub(r'\n{3,}', '\n\n', content)  # שורות ריקות מרובות
        content = re.sub(r' {2,}', ' ', content)  # רווחים מרובים
        content = re.sub(r'^\s*=+.*?=+\s*$', '', content, flags=re.MULTILINE)  # כותרות ויקי

        return content

    def identify_headers(self, content):
        """
        זיהוי וסימון כותרות (זהה לחלוטין לתוכנית המדגם)
        """
        lines = content.split('\n')
        processed_lines = []

        for i, line in enumerate(lines):
            line = line.strip()

            # זיהוי כותרת פשוט
            if (line and len(line) < 100 and
                    i < len(lines) - 1 and
                    len(lines[i + 1].strip()) > 50 and
                    line.count('.') <= 1 and
                    line.count(',') <= 2):

                # בדיקות נוספות לוודא שזו כותרת
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
        """
        ניקוי תוכן (זהה לחלוטין לתוכנית המדגם)
        """
        try:
            # הסרת תבניות
            templates_to_remove = []
            for template in wikicode.filter_templates():
                template_name = str(template.name).strip().lower()
                # תבניות להסרה מלאה
                remove_patterns = [
                    'cite', 'צ-', 'הערה', 'מקור', 'reflist', 'מקורות',
                    'ציון', 'ref', 'citation', 'web', 'news', 'book', 'journal'
                ]
                if any(pattern in template_name for pattern in remove_patterns):
                    templates_to_remove.append(template)

            # הסרת התבניות
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
                        # זהה בדיוק לתוכנית המדגם - ללא .strip()
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
            # במקרה של שגיאה
            basic_clean = str(wikicode)[:2000]
            return self.normalize_text_for_training(basic_clean)

    def count_words(self, text):
        """
        ספירת מילים
        """
        if not text:
            return 0
        # ספירת מילים פשוטה - חלוקה לפי רווחים
        words = text.split()
        return len(words)

    def count_bytes(self, text):
        """
        ספירת בייטים ב-UTF-8
        """
        if not text:
            return 0
        return len(text.encode('utf-8'))

    def process_article(self, title, raw_wikitext):
        """
        עיבוד ערך יחיד ליצירת פריט JSONL
        """
        try:
            # עיבוד הויקי-טקסט
            wikicode = mwparserfromhell.parse(raw_wikitext)
            processed_text = self.clean_content(wikicode)

            # וידוא שהטקסט לא ריק מדי
            if not processed_text or len(processed_text) < 100:
                return None

            # יצירת פריט JSONL
            article_item = {
                "text": processed_text,
                "word_count": self.count_words(processed_text),
                "byte_count": self.count_bytes(processed_text)
            }

            return article_item

        except Exception as e:
            print("Error processing article '" + title + "': " + str(e))
            return None

    def is_valid_article(self, page_elem):
        """
        בדיקה אם הדף תקין לעיבוד (זהה לתוכנית המדגם)
        """
        # רק namespace 0 (ערכים ראשיים)
        namespace_elem = page_elem.find('.//{*}ns')
        if namespace_elem is None or namespace_elem.text != '0':
            return False

        # בדיקת תוכן בסיסית
        revision = page_elem.find('.//{*}revision')
        if revision is None:
            return False

        text_elem = revision.find('.//{*}text')
        if text_elem is None or not text_elem.text:
            return False

        # בדיקת אורך מינימלי
        if len(text_elem.text) < 500:
            return False

        # דילוג על הפניות
        if text_elem.text.strip().startswith('#REDIRECT'):
            return False

        return True

    def process_dump(self):
        """
        עיבוד הדאמפ וכתיבה ל-JSONL
        """
        print("🚀 מתחיל עיבוד ויקיפדיה ל-JSONL")
        print("📂 דאמפ: " + self.dump_path)
        print("💾 פלט: " + self.output_file)
        print("🎯 מטרה: " + str(self.max_articles) + " ערכים")
        print("=" * 60)

        processed_count = 0
        scanned_count = 0

        # פתיחת קובץ הפלט
        with open(self.output_file, 'w', encoding='utf-8') as output:

            # עיבוד הדאמפ
            with bz2.open(self.dump_path, 'rt', encoding='utf-8') as dump_file:

                with tqdm(total=self.max_articles, desc="🔄 עיבוד ערכים", unit="articles") as pbar:

                    for event, elem in ET.iterparse(dump_file, events=('start', 'end')):
                        if event == 'end' and elem.tag.endswith('page'):
                            scanned_count += 1

                            # הדפסת התקדמות כל 1000 דפים
                            if scanned_count % 1000 == 0:
                                pbar.set_postfix_str(
                                    "סרקתי: " + str(scanned_count) + ", עיבדתי: " + str(processed_count))

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
                                    # כתיבה ל-JSONL
                                    json.dump(article_item, output, ensure_ascii=False)
                                    output.write('\n')

                                    processed_count += 1
                                    pbar.update(1)
                                    pbar.set_postfix_str("נוכחי: " + title[:30] + "...")

                                    # הפסקה אם הגענו למטרה
                                    if processed_count >= self.max_articles:
                                        break

                            elem.clear()

                            # הפסקה אם הגענו למטרה
                            if processed_count >= self.max_articles:
                                break

        # סיכום
        print("\n" + "=" * 60)
        print("🎉 עיבוד הושלם!")
        print("📊 סטטיסטיקות:")
        print("   דפים שנסרקו: " + str(scanned_count))
        print("   ערכים שעובדו: " + str(processed_count))
        print("   שיעור הצלחה: " + str(round((processed_count / scanned_count) * 100, 2)) + "%")

        # ניתוח הקובץ שנוצר
        self.analyze_output()

    def analyze_output(self):
        """
        ניתוח הקובץ שנוצר
        """
        print("\n📊 ניתוח הקובץ: " + self.output_file)
        print("-" * 40)

        total_items = 0
        total_words = 0
        total_bytes = 0
        word_counts = []
        byte_counts = []

        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        total_items += 1
                        total_words += item['word_count']
                        total_bytes += item['byte_count']
                        word_counts.append(item['word_count'])
                        byte_counts.append(item['byte_count'])

            print("📝 סה\"כ ערכים: " + str(total_items))
            print("📏 סה\"כ מילים: " + str(total_words))
            print("💾 סה\"כ בייטים: " + str(total_bytes) + " (" + str(round(total_bytes / 1024 / 1024, 2)) + " MB)")

            if word_counts:
                print("📊 ממוצע מילים לערך: " + str(round(sum(word_counts) / len(word_counts), 1)))
                print("📊 ממוצע בייטים לערך: " + str(round(sum(byte_counts) / len(byte_counts), 1)))
                print("📊 ערך הקצר ביותר: " + str(min(word_counts)) + " מילים")
                print("📊 ערך הארוך ביותר: " + str(max(word_counts)) + " מילים")

            # דוגמה מהקובץ
            print("\n📄 דוגמה מהקובץ:")
            with open(self.output_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if first_line:
                    example = json.loads(first_line)
                    print("   מילים: " + str(example['word_count']))
                    print("   בייטים: " + str(example['byte_count']))
                    print("   טקסט (100 תווים): " + example['text'][:100] + "...")

        except Exception as e:
            print("❌ שגיאה בניתוח: " + str(e))


def main():
    """
    הפעלה ראשית
    """
    print("🎯 עיבוד דגימה של ויקיפדיה ל-JSONL")
    print("=" * 60)

    # הגדרות
    dump_path = r'C:\Users\Dan Revital\OneDrive\Documents\gepeta\data\wikipedia\hewiki-latest-pages-articles.xml.bz2'
    output_file = 'wikipedia_he_sample_100.jsonl'
    max_articles = 100

    # יצירת מעבד
    processor = WikipediaJSONLProcessor(
        dump_path=dump_path,
        output_file=output_file,
        max_articles=max_articles
    )

    # הפעלת העיבוד
    processor.process_dump()

    print("\n✅ הקובץ נוצר: " + output_file)
    print("💡 אפשר לבדוק אותו עם:")
    print("   head -n 3 " + output_file)


if __name__ == "__main__":
    main()