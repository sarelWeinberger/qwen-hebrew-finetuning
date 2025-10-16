#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikipedia Hebrew Text Cleaner - Core Module
============================================

מודול מרכזי לניקוי טקסט ויקיפדיה עברית.
מכיל את כל לוגיקת הניקוי המשותפת לכל ה-use cases השונים.
"""

import re
import ipaddress
import mwparserfromhell
from typing import Optional, Dict, Callable, Any
from collections import defaultdict


class WikipediaTextCleaner:
    """מחלקה מרכזית לניקוי טקסט ויקיפדיה עברית"""

    def __init__(self, example_callback: Optional[Callable[[str, str, str], None]] = None):
        """
        אתחול המנקה

        Args:
            example_callback: פונקציה לשמירת דוגמאות (category, raw_text, clean_text)
                             אם None, לא נשמרות דוגמאות
        """
        self.example_callback = example_callback
        self.stats = defaultdict(int)  # סטטיסטיקות לניקוי

    def _save_example(self, category: str, raw_text: str, clean_text: str):
        """שמירת דוגמה אם קיים callback"""
        if self.example_callback:
            self.example_callback(category, raw_text, clean_text)
            self.stats[f"{category}_examples"] += 1

    def clean_html_escape_codes(self, text: str) -> str:
        """כלל 1: החלפת HTML escape codes"""
        original_text = text

        # החלפת escape codes שונים
        text = text.replace('&quot;', '"')
        text = text.replace('&#34;', '"')
        text = text.replace('&#39;', "'")

        # שמירת דוגמה אם השתנה משהו
        if text != original_text:
            self._save_example("html_escape_codes", original_text, text)
            self.stats["html_escape_codes_applied"] += 1

        return text

    def clean_newlines_and_spaces(self, text: str) -> str:
        """כלל 2: טיפול בשורות חדשות ורווחים"""
        original_text = text

        # הסרת carriage return
        text = text.replace('\r', '')

        # החלפת יותר מ-3 שורות חדשות רצופות במקסימום 3
        text = re.sub(r'\n{4,}', '\n\n\n', text)

        # שמירת דוגמה אם השתנה משהו
        if text != original_text:
            self._save_example("cr_and_3_newlines", original_text, text)
            self.stats["newlines_applied"] += 1

        return text

    def clean_multiple_spaces(self, text: str) -> str:
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
            self._save_example("multiple_spaces", original_text, text)
            self.stats["multiple_spaces_applied"] += 1

        return text

    def clean_whitespace_start_end(self, text: str) -> str:
        """כלל 4: הסרת רווחים מתחילת וסוף הטקסט"""
        original_text = text
        text = text.strip()

        # שמירת דוגמה אם השתנה משהו
        if text != original_text:
            self._save_example("start_end_white_space", original_text, text)
            self.stats["whitespace_trim_applied"] += 1

        return text

    def is_localhost_ip(self, ip_str: str) -> bool:
        """בדיקה אם IP הוא localhost (127.0.0.0/8)"""
        try:
            ip = ipaddress.ip_address(ip_str)
            localhost_network = ipaddress.ip_network('127.0.0.0/8')
            return ip in localhost_network
        except:
            return False

    def clean_pii(self, text: str) -> str:
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
            self._save_example("PII", original_text, text)
            self.stats["pii_applied"] += 1

        return text

    def clean_empty_bullet_lines(self, text: str) -> str:
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
            self._save_example("empty_line_with_bullet", original_text, text)
            self.stats["bullet_lines_applied"] += 1

        return text

    def clean_separator_lines(self, text: str) -> str:
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
            self._save_example("multiple_hyphens", original_text, text)
            self.stats["separator_lines_applied"] += 1

        return text

    def clean_css_from_tables(self, text: str) -> str:
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
            self._save_example("CSS", original_text, text)
            self.stats["css_applied"] += 1

        return text

    def convert_tables_to_markdown(self, text: str) -> str:
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
            self._save_example("tables", original_text, text)
            self.stats["tables_applied"] += 1

        return text

    def clean_wiki_templates_and_markup(self, text: str) -> str:
        """כלל 11a: הסרת תבניות ו-markup של ויקי"""
        original_text = text

        # הסרת תבניות מורכבות עם תבניות מקוננות {{}}
        # עושה זאת מספר פעמים כדי להתמודד עם קינון עמוק
        for _ in range(5):  # מקסימום 5 איטרציות
            old_text = text
            text = re.sub(r'\{\{[^{}]*\}\}', '', text)
            if text == old_text:  # אם לא השתנה, מפסיק
                break

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
            self._save_example("wiki_markup", original_text, text)
            self.stats["markup_applied"] += 1

        return text

    def clean_wiki_media_descriptions(self, text: str) -> str:
        """כלל 11b: הסרת תיאורי מדיה ותמונות - גרסה מתוקנת"""
        original_text = text

        # 1. הסרת קישורי קבצים/תמונות מלאים לפני הכל
        media_link_patterns = [
            r'\[\[קובץ:[^\]]*\]\]',
            r'\[\[File:[^\]]*\]\]',
            r'\[\[Image:[^\]]*\]\]',
            r'\[\[תמונה:[^\]]*\]\]',
            r'\[\[media:[^\]]*\]\]',
            r'\[\[מדיה:[^\]]*\]\]'
        ]

        for pattern in media_link_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

        # 2. הסרת תיאורי מיקום תמונות - דפוסים משופרים
        location_keywords = [
            'שמאל', 'ימין', 'מרכז', 'ממוזער', 'מוקטן', 'מסגרת',
            'thumb', 'thumbnail', 'frame', 'framed', 'frameless',
            'left', 'right', 'center', 'centre', 'none',
            'upright', 'border'
        ]

        # הסרת תיאורי מיקום עם סימן |
        for keyword in location_keywords:
            # דפוסים שונים לתיאורי מיקום
            patterns = [
                rf'\|{keyword}\|',  # |שמאל|
                rf'\|{keyword}\b[^|]*\|',  # |שמאל עם פרמטרים|
                rf'^{keyword}\|',  # שמאל| בתחילת שורה
                rf'\s{keyword}\|',  # רווח שמאל|
                rf'\|{keyword}$',  # |שמאל בסוף
                rf'\|{keyword}\s',  # |שמאל רווח
            ]

            for pattern in patterns:
                text = re.sub(pattern, '|', text, flags=re.IGNORECASE | re.MULTILINE)

        # 3. הסרת הגדרות גודל תמונות
        size_patterns = [
            r'\b\d+px\b',  # 300px
            r'\|\d+px\|',  # |300px|
            r'\|\d+px\b',  # |300px
            r'\b\d+x\d+px\b',  # 300x200px
            r'\|\d+x\d+px\|',  # |300x200px|
            r'width\s*=\s*["\']?\d+["\']?',  # width="300"
            r'height\s*=\s*["\']?\d+["\']?',  # height="200"
        ]

        for pattern in size_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # 4. הסרת תיאורים בסוגריים עם מונחי מדיה
        media_terms = ['px', 'ממוזער', 'thumb', 'frame', 'שמאל', 'ימין', 'מרכז', 'תמונה', 'קובץ']
        for term in media_terms:
            pattern = rf'\([^)]*{term}[^)]*\)'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # 5. הסרת פרמטרי תמונה נוספים
        image_params = [
            r'\|alt\s*=[^|]*',  # alt text
            r'\|caption\s*=[^|]*',  # כיתוב
            r'\|class\s*=[^|]*',  # CSS class
            r'\|link\s*=[^|]*',  # קישור
            r'\|page\s*=[^|]*',  # עמוד
            r'\|lang\s*=[^|]*',  # שפה
        ]

        for pattern in image_params:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # 6. הסרת הפניות "ראו" - תמיכה מורחבת
        see_patterns = [
            r'ראו [A-Za-z\s,א-ת\.\:\-]+',
            r'ראה [A-Za-z\s,א-ת\.\:\-]+',
            r'see [A-Za-z\s,\.\:\-]+',
            r'See [A-Za-z\s,\.\:\-]+',
            r'ראו גם [A-Za-z\s,א-ת\.\:\-]+',
            r'ראה גם [A-Za-z\s,א-ת\.\:\-]+',
            r'see also [A-Za-z\s,\.\:\-]+',
            r'See also [A-Za-z\s,\.\:\-]+',
            r'להרחבה ראו [A-Za-z\s,א-ת\.\:\-]+',
            r'לפרטים נוספים ראו [A-Za-z\s,א-ת\.\:\-]+',
            r'for more details see [A-Za-z\s,\.\:\-]+',
        ]

        for pattern in see_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # 7. הסרת שורות "ראו גם" שלמות
        see_also_line_patterns = [
            r'^ראו גם\s*[:]*\s*$',
            r'^ראה גם\s*[:]*\s*$',
            r'^see also\s*[:]*\s*$',
            r'^See also\s*[:]*\s*$',
            r'^\s*ראו גם\s*[:]*.*$',
            r'^\s*see also\s*[:]*.*$',
        ]

        for pattern in see_also_line_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

        # 8. הסרת תיאורי gallery ומדיה מורכבים
        gallery_patterns = [
            r'<gallery[^>]*>.*?</gallery>',
            r'\{\{gallery[^}]*\}\}',
            r'<imagemap[^>]*>.*?</imagemap>',
        ]

        for pattern in gallery_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

        # 9. ניקוי סימני | מיותרים שנותרו
        text = re.sub(r'\|\s*\|', '|', text)  # ||
        text = re.sub(r'^\|', '', text, flags=re.MULTILINE)  # | בתחילת שורה
        text = re.sub(r'\|$', '', text, flags=re.MULTILINE)  # | בסוף שורה
        text = re.sub(r'\|\s*$', '', text, flags=re.MULTILINE)  # | ורווחים בסוף שורה

        # 10. ניקוי שורות ריקות עודפות שנוצרו
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'\n{4,}', '\n\n\n', text)

        # 11. ניקוי רווחים מיותרים (רק רווחים וטאבים, לא שורות חדשות)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)

        # 12. הסרת שורות ריקות שנותרו עם רק סימני פיסוק
        text = re.sub(r'^\s*[,\.;:]+\s*$', '', text, flags=re.MULTILINE)

        # שמירת דוגמה אם השתנה משהו
        if text != original_text:
            self._save_example("wiki_tags_and_media_descriptions", original_text, text)
            self.stats["media_descriptions_applied"] += 1

        return text

    def clean_wiki_headers(self, text: str) -> str:
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
            self._save_example("wiki_headers", original_text, text)
            self.stats["headers_applied"] += 1

        return text

    def clean_wiki_citations(self, text: str) -> str:
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
            self._save_example("wiki_citations", original_text, text)
            self.stats["citations_applied"] += 1

        return text

    def check_redirecting_article(self, raw_wikitext: str) -> bool:
        """כלל 11f: בדיקה אם המאמר הוא הפניה"""
        is_redirect = raw_wikitext.strip().startswith('#REDIRECT') or raw_wikitext.strip().startswith('#הפניה')

        if is_redirect:
            self._save_example("wiki_redirecting_articles", raw_wikitext, "[REDIRECTING ARTICLE - EXCLUDED]")
            self.stats["redirects_found"] += 1

        return is_redirect

    def clean_article(self, title: str, raw_wikitext: str, min_length: int = 100) -> Optional[str]:
        """
        מנקה ערך ויקיפדיה שלם

        Args:
            title: כותרת הערך
            raw_wikitext: ויקי-טקסט גולמי
            min_length: אורך מינימלי של טקסט נקי (ברירת מחדל: 100)

        Returns:
            טקסט נקי או None אם הערך נפסל
        """
        self.stats["articles_processed"] += 1

        # בדיקה אם זה מאמר מפנה (כלל 11f)
        if self.check_redirecting_article(raw_wikitext):
            self.stats["articles_rejected_redirect"] += 1
            return None

        # המרה לאובייקט wikicode
        try:
            wikicode = mwparserfromhell.parse(raw_wikitext)
            text = str(wikicode)
        except:
            text = raw_wikitext

        # הפעלת כל הכללים בסדר המעודכן
        text = self.clean_html_escape_codes(text)  # כלל 1
        text = self.clean_newlines_and_spaces(text)  # כלל 2
        text = self.clean_multiple_spaces(text)  # כלל 3
        text = self.clean_whitespace_start_end(text)  # כלל 4
        text = self.clean_pii(text)  # כלל 6
        text = self.clean_empty_bullet_lines(text)  # כלל 7
        text = self.clean_separator_lines(text)  # כלל 8
        text = self.clean_css_from_tables(text)  # כלל 9
        text = self.convert_tables_to_markdown(text)  # כלל 10
        text = self.clean_wiki_citations(text)  # כלל 11e - מועבר לפני markup
        text = self.clean_wiki_media_descriptions(text)  # כלל 11b - מועבר לפני markup
        text = self.clean_wiki_templates_and_markup(text)  # כלל 11a - אחרי media
        text = self.clean_wiki_headers(text)  # כלל 11d

        # בדיקה סופית של איכות
        if not text or len(text) < min_length:
            self.stats["articles_rejected_too_short"] += 1
            return None

        self.stats["articles_cleaned_successfully"] += 1
        return text

    def clean_text_only(self, text: str, min_length: int = 100) -> Optional[str]:
        """
        מנקה טקסט בלבד (ללא ויקי-קוד parsing)

        Args:
            text: טקסט גולמי
            min_length: אורך מינימלי של טקסט נקי

        Returns:
            טקסט נקי או None אם נפסל
        """
        self.stats["text_only_processed"] += 1

        # הפעלת כל הכללים בסדר המעודכן (ללא בדיקת redirect כי זה לא ויקי-טקסט מלא)
        text = self.clean_html_escape_codes(text)  # כלל 1
        text = self.clean_newlines_and_spaces(text)  # כלל 2
        text = self.clean_multiple_spaces(text)  # כלל 3
        text = self.clean_whitespace_start_end(text)  # כלל 4
        text = self.clean_pii(text)  # כלל 6
        text = self.clean_empty_bullet_lines(text)  # כלל 7
        text = self.clean_separator_lines(text)  # כלל 8
        text = self.clean_css_from_tables(text)  # כלל 9
        text = self.convert_tables_to_markdown(text)  # כלל 10
        text = self.clean_wiki_citations(text)  # כלל 11e - מועבר לפני markup
        text = self.clean_wiki_media_descriptions(text)  # כלל 11b - מועבר לפני markup
        text = self.clean_wiki_templates_and_markup(text)  # כלל 11a - אחרי media
        text = self.clean_wiki_headers(text)  # כלל 11d

        # בדיקה סופית של איכות
        if not text or len(text) < min_length:
            self.stats["text_only_rejected_too_short"] += 1
            return None

        self.stats["text_only_cleaned_successfully"] += 1
        return text

    def get_stats(self) -> Dict[str, int]:
        """החזרת סטטיסטיקות הניקוי"""
        return dict(self.stats)

    def reset_stats(self):
        """איפוס סטטיסטיקות"""
        self.stats.clear()


# פונקציות עזר למקרים פשוטים
def clean_wikipedia_article(title: str, raw_wikitext: str, min_length: int = 100) -> Optional[str]:
    """
    פונקציה פשוטה לניקוי ערך ויקיפדיה ללא שמירת דוגמאות

    Args:
        title: כותרת הערך
        raw_wikitext: ויקי-טקסט גולמי
        min_length: אורך מינימלי של טקסט נקי

    Returns:
        טקסט נקי או None אם הערך נפסל
    """
    cleaner = WikipediaTextCleaner()
    return cleaner.clean_article(title, raw_wikitext, min_length)


def clean_text(text: str, min_length: int = 100) -> Optional[str]:
    """
    פונקציה פשוטה לניקוי טקסט ללא שמירת דוגמאות

    Args:
        text: טקסט גולמי
        min_length: אורך מינימלי של טקסט נקי

    Returns:
        טקסט נקי או None אם נפסל
    """
    cleaner = WikipediaTextCleaner()
    return cleaner.clean_text_only(text, min_length)


# פונקציות עזר למטריקות
def count_words(text: str) -> int:
    """ספירת מילים"""
    if not text:
        return 0
    return len(text.split())


def count_bytes(text: str) -> int:
    """ספירת בייטים ב-UTF-8"""
    if not text:
        return 0
    return len(text.encode('utf-8'))