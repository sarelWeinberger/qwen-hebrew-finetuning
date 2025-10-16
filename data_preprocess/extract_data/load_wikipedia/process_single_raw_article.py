#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import mwparserfromhell
import re
import bz2
import ipaddress
from pathlib import Path


def find_article_in_dump(dump_path, article_title):
    """
    מוצא ערך ספציפי בדאמפ ויקיפדיא
    """
    print(f"🔍 מחפש ערך: '{article_title}' בדאמפ...")
    print(f"📂 דאמפ: {dump_path}")
    print("⏳ זה עלול לקחת זמן...")

    scanned_pages = 0

    with bz2.open(dump_path, 'rt', encoding='utf-8') as file:
        for event, elem in ET.iterparse(file, events=('start', 'end')):
            if event == 'end' and elem.tag.endswith('page'):
                scanned_pages += 1

                # הדפסת התקדמות כל 10,000 דפים
                if scanned_pages % 10000 == 0:
                    print(f"   סרקתי {scanned_pages:,} דפים...")

                # חילוץ הכותרת
                title_elem = elem.find('.//{*}title')
                if title_elem is not None and title_elem.text == article_title:

                    print(f"✅ נמצא ערך: '{article_title}' אחרי {scanned_pages:,} דפים!")

                    # חילוץ הטקסט
                    revision = elem.find('.//{*}revision')
                    text_elem = revision.find('.//{*}text') if revision is not None else None

                    if text_elem is not None and text_elem.text:
                        raw_wikitext = text_elem.text
                        print(f"📏 אורך ויקי-טקסט גולמי: {len(raw_wikitext):,} תווים")

                        elem.clear()
                        return raw_wikitext
                    else:
                        print(f"❌ לא נמצא תוכן בערך")
                        elem.clear()
                        return None

                elem.clear()

    print(f"❌ לא נמצא ערך: '{article_title}' (סרקתי {scanned_pages:,} דפים)")
    return None


class SingleArticleProcessor:
    """מחלקה לעיבוד ערך יחיד - בדיוק כמו בקוד המעודכן"""

    def clean_html_escape_codes(self, text):
        """כלל 1: החלפת HTML escape codes"""
        # החלפת escape codes שונים
        text = text.replace('&quot;', '"')
        text = text.replace('&#34;', '"')
        text = text.replace('&#39;', "'")
        return text

    def clean_newlines_and_spaces(self, text):
        """כלל 2: טיפול בשורות חדשות ורווחים"""
        # הסרת carriage return
        text = text.replace('\r', '')
        # החלפת יותר מ-3 שורות חדשות רצופות במקסימום 3
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        return text

    def clean_multiple_spaces(self, text):
        """כלל 3: טיפול ברווחים מרובים"""

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
        return text

    def clean_whitespace_start_end(self, text):
        """כלל 4: הסרת רווחים מתחילת וסוף הטקסט"""
        return text.strip()

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
        return text

    def clean_empty_bullet_lines(self, text):
        """כלל 7: הסרת שורות ריקות עם bullet points"""
        # הסרת שורות שמכילות רק bullet points ורווחים
        bullet_pattern = r'^\s*[•●■◦▪◆]+\s*$'
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            if not re.match(bullet_pattern, line):
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def clean_separator_lines(self, text):
        """כלל 8: הסרת קווי הפרדה ארוכים"""
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

        return '\n'.join(cleaned_lines)

    def clean_css_from_tables(self, text):
        """כלל 9: הסרת CSS מטבלאות"""
        # הסרת style attributes
        text = re.sub(r'style="[^"]*"', '', text)
        text = re.sub(r'cellspacing="[^"]*"', '', text)
        text = re.sub(r'cellpadding="[^"]*"', '', text)
        text = re.sub(r'class="[^"]*"', '', text)
        text = re.sub(r'width="[^"]*"', '', text)
        text = re.sub(r'height="[^"]*"', '', text)
        return text

    def convert_tables_to_markdown(self, text):
        """כלל 10: המרת טבלאות ויקי למרקדאון"""

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
        return text

    def clean_wiki_templates_and_markup(self, text):
        """כלל 11a,b: הסרת תבניות ו-markup של ויקי"""
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

        return text

    def clean_wiki_media_descriptions(self, text):
        """כלל 11b: הסרת תיאורי מדיה ותמונות"""

        # הסרת תיאורי קבצי מדיה (File:, קובץ:, Image:)
        media_patterns = [
            r'\[\[קובץ:.*?\]\]',
            r'\[\[File:.*?\]\]',
            r'\[\[Image:.*?\]\]',
            r'\[\[תמונה:.*?\]\]'
        ]

        for pattern in media_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

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

        return text

    def clean_wiki_headers(self, text):
        """כלל 11d: המרת כותרות ויקי לפורמט מרקדאון"""

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

        return text

    def clean_wiki_citations(self, text):
        """כלל 11e: הסרת citations"""
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

        return text

    def check_redirecting_article(self, raw_wikitext):
        """כלל 11f: בדיקה אם המאמר הוא הפניה"""
        is_redirect = raw_wikitext.strip().startswith('#REDIRECT') or raw_wikitext.strip().startswith('#הפניה')
        return is_redirect

    def apply_all_cleaning_rules(self, title, raw_wikitext):
        """הפעלת כל כללי הניקיון על טקסט - בדיוק כמו בקוד המעודכן"""
        # בדיקה אם זה מאמר מפנה (כלל 11f)
        if self.check_redirecting_article(raw_wikitext):
            return None

        # המרה לאובייקט wikicode
        try:
            wikicode = mwparserfromhell.parse(raw_wikitext)
            text = str(wikicode)
        except:
            text = raw_wikitext

        # הפעלת כל הכללים בדיוק באותו סדר
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


def process_with_updated_method(raw_wikitext, article_title):
    """
    עיבוד עם השיטה המעודכנת (בדיוק כמו בקוד המעודכן)
    """
    print(f"⚙️ מעבד את הויקי-טקסט...")

    processor = SingleArticleProcessor()

    # עיבוד הערך
    cleaned_text = processor.apply_all_cleaning_rules(article_title, raw_wikitext)

    if cleaned_text:
        print(f"✅ עיבוד הושלם בהצלחה")
        print(f"📄 אורך אחרי עיבוד: {len(cleaned_text):,} תווים")
        return cleaned_text
    else:
        print(f"❌ הערך נפסל בעיבוד (ייתכן שהוא הפניה או קצר מדי)")
        return None


def main():
    """
    הפעלה ראשית
    """
    print("🎯 מוצא ומעבד ערך מהדאמפ עם השיטה המעודכנת")
    print("=" * 60)

    # הגדרות
    dump_path = r'C:\Users\Dan Revital\OneDrive\Documents\gepeta\data\wikipedia\hewiki-latest-pages-articles.xml.bz2'
    article_title = 'הבינום של ניוטון'

    print(f"🔍 מחפש ערך: '{article_title}'")
    print(f"📂 בדאמפ: {dump_path}")
    print()

    # שלב 1: מציאת הערך
    raw_wikitext = find_article_in_dump(dump_path, article_title)

    if not raw_wikitext:
        print(f"❌ לא נמצא הערך '{article_title}'")
        return

    print()

    # שלב 2: עיבוד עם השיטה המעודכנת
    print("=" * 60)
    processed_content = process_with_updated_method(raw_wikitext, article_title)

    if not processed_content:
        print("❌ העיבוד נכשל")
        return

    # שלב 3: שמירת התוצאות
    print("=" * 60)
    print("💾 שומר תוצאות...")

    # שמירת הטקסט הגולמי
    raw_filename = f"{article_title.replace(' ', '_')}_raw_wikitext.txt"
    with open(raw_filename, 'w', encoding='utf-8') as f:
        f.write(raw_wikitext)
    print(f"✅ ויקי-טקסט גולמי נשמר: {raw_filename}")

    # שמירת הטקסט המעובד
    processed_filename = f"{article_title.replace(' ', '_')}_updated_processed.txt"
    with open(processed_filename, 'w', encoding='utf-8') as f:
        f.write(processed_content)
    print(f"✅ טקסט מעובד נשמר: {processed_filename}")

    # סיכום
    print("\n" + "=" * 60)
    print("🎉 הושלם בהצלחה!")
    print(f"📊 סיכום:")
    print(f"   ויקי-טקסט גולמי: {len(raw_wikitext):,} תווים")
    print(f"   טקסט מעובד: {len(processed_content):,} תווים")
    print(f"   דחיסה: {(1 - len(processed_content) / len(raw_wikitext)) * 100:.1f}%")

    print(f"\n📄 דוגמה מהטקסט המעובד (200 תווים ראשונים):")
    print(f"   {processed_content[:200]}...")


if __name__ == "__main__":
    main()