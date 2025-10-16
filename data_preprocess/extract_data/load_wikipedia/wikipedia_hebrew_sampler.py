import xml.etree.ElementTree as ET
import mwparserfromhell
import json
import re
import random
from pathlib import Path
import bz2
from tqdm import tqdm


class WikipediaSampleProcessor:
    def __init__(self, dump_path, output_file, sample_size=100):
        self.dump_path = dump_path
        self.output_file = output_file
        self.sample_size = sample_size
        self.articles = []
        self.valid_pages = []

    def collect_valid_pages(self, max_scan=50000):
        """אוסף דפים תקינים לדגימה עם פרוגרס באר"""
        print(f"🔍 Scanning dump for valid pages (max {max_scan})...")

        with tqdm(
                total=max_scan,
                desc="📖 Scanning pages",
                unit="pages",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        ) as pbar:

            with bz2.open(self.dump_path, 'rt', encoding='utf-8') as file:
                scanned = 0
                for event, elem in ET.iterparse(file, events=('start', 'end')):
                    if event == 'end' and elem.tag.endswith('page'):
                        if self.is_valid_page(elem):
                            page_data = self.extract_basic_info(elem)
                            if page_data:
                                self.valid_pages.append(page_data)

                        elem.clear()
                        scanned += 1

                        # עדכון פרוגרס באר
                        pbar.update(1)
                        valid_ratio = len(self.valid_pages) / scanned * 100 if scanned > 0 else 0
                        pbar.set_postfix_str(f"Valid: {len(self.valid_pages)} ({valid_ratio:.1f}%)")

                        # הפסק אחרי סריקה מספקת או אם מצאנו מספיק
                        if scanned >= max_scan or len(self.valid_pages) >= self.sample_size * 5:
                            break

        print(f"✅ Found {len(self.valid_pages)} valid pages out of {scanned} scanned")

    def is_valid_page(self, page_elem):
        """בדיקה מהירה אם הדף תקין"""
        # רק namespace 0 (ערכים ראשיים)
        namespace = self.get_text(page_elem, 'ns')
        if namespace != '0':
            return False

        # בדיקת תוכן בסיסית
        revision = page_elem.find('.//{*}revision')
        if revision is None:
            return False

        text = self.get_text(revision, 'text')
        if not text or len(text) < 500:
            return False

        if text.strip().startswith('#REDIRECT'):
            return False

        return True

    def extract_basic_info(self, page_elem):
        """חילוץ מידע בסיסי לדגימה"""
        title = self.get_text(page_elem, 'title')
        page_id = self.get_text(page_elem, 'id')

        revision = page_elem.find('.//{*}revision')
        text = self.get_text(revision, 'text')

        return {
            'id': page_id,
            'title': title,
            'text': text,
            'length': len(text)
        }

    def get_text(self, element, tag_name):
        """חילוץ טקסט מאלמנט XML"""
        elem = element.find(f'.//{{{element.tag.split("}")[0][1:]}}}{tag_name}')
        return elem.text if elem is not None and elem.text else ""

    def create_sample(self):
        """יצירת דגימה אקראית עם פרוגרס באר"""
        if len(self.valid_pages) < self.sample_size:
            print(f"⚠️ Warning: Only {len(self.valid_pages)} valid pages found, using all of them")
            sample = self.valid_pages
        else:
            sample = random.sample(self.valid_pages, self.sample_size)

        print(f"⚙️ Processing {len(sample)} articles...")

        # פרוגרס באר עם מידע מפורט
        successful = 0
        failed = 0

        for page_data in tqdm(sample, desc="📄 Processing articles", unit="articles"):
            try:
                processed = self.process_article(page_data)
                if processed:
                    self.articles.append(processed)
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                failed += 1
                tqdm.write(f"❌ Error processing '{page_data['title'][:30]}...': {str(e)[:50]}")
                continue

        print(f"✅ Successfully processed: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Success rate: {successful / (successful + failed) * 100:.1f}%")

    def process_article(self, page_data):
        """עיבוד ערך יחיד"""
        try:
            # עיבוד תוכן ויקי
            processed = self.process_wikitext(page_data['text'])

            if not processed['content'] or len(processed['content']) < 200:
                return None

            return {
                'id': page_data['id'],
                'title': page_data['title'],
                'content': processed['content'],
                'summary': processed['summary'],
                'categories': processed['categories'],
                'infobox': processed['infobox'],
                'tables': processed['tables'],
                'original_length': page_data['length'],
                'processed_length': len(processed['content']),
                'quality_score': self.calculate_quality(processed)
            }

        except Exception as e:
            print(f"Error in process_article: {e}")
            return None

    def process_wikitext(self, wikitext):
        """עיבוד טקסט ויקי למבנה נקי"""
        try:
            wikicode = mwparserfromhell.parse(wikitext)
        except Exception as e:
            print(f"Error parsing wikitext: {e}")
            return {
                'content': self.normalize_text_for_training(wikitext[:1000]),
                'summary': '',
                'categories': [],
                'infobox': '',
                'tables': []
            }

        # חילוץ רכיבים שונים
        categories = self.extract_categories(wikicode)
        infobox = self.extract_infobox(wikicode)
        tables = self.extract_tables(wikicode)

        # ניקוי הטקסט הראשי
        content = self.clean_content(wikicode)
        summary = self.extract_summary(content)

        return {
            'content': content,
            'summary': summary,
            'categories': categories,
            'infobox': infobox,
            'tables': tables
        }

    def normalize_text_for_training(self, text):
        """
        מנרמל טקסט לאימון מודל שפה:
        - מחק כל שורות חדשות
        - מחק טאבים
        - רווחים מרובים → רווח יחיד
        - שמור רק את התוכן האמיתי
        """
        if not text or not isinstance(text, str):
            return text

        # מחק שורות חדשות ו-\r
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')

        # מחק כל תווי בריחה אחרים
        text = text.replace('\\n', ' ')
        text = text.replace('\\t', ' ')
        text = text.replace('\\r', ' ')
        text = text.replace('\\"', '"')
        text = text.replace("\\'", "'")
        text = text.replace('\\\\', '\\')

        # רווחים מרובים → רווח יחיד
        text = re.sub(r'\s+', ' ', text)

        # נקה רווחים בהתחלה/סוף
        text = text.strip()

        return text

    def clean_content(self, wikicode):
        """ניקוי תוכן - גרסה מנורמלת לאימון"""
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
                        wikicode.replace(tag, f"[נוסחה: {tag.contents}]")
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

            # נרמול הטקסט לאימון - זה החלק החדש!
            content = self.normalize_text_for_training(content)

            return content

        except Exception as e:
            print(f"Error in clean_content: {e}")
            # במקרה של שגיאה
            basic_clean = str(wikicode)[:2000]
            return self.normalize_text_for_training(basic_clean)

    def apply_regex_cleaning(self, content):
        """ניקוי regex מרוכז"""
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

        # ניקוי כללי - נשאיר את זה כי הנרמול יטפל בהמשך
        content = re.sub(r'\n{3,}', '\n\n', content)  # שורות ריקות מרובות
        content = re.sub(r' {2,}', ' ', content)  # רווחים מרובים
        content = re.sub(r'^\s*=+.*?=+\s*$', '', content, flags=re.MULTILINE)  # כותרות ויקי

        return content

    def identify_headers(self, content):
        """זיהוי וסימון כותרות"""
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
                    processed_lines.append(f"## {line}")
                else:
                    processed_lines.append(line)
            else:
                processed_lines.append(line)

        return '\n'.join(processed_lines)

    def extract_summary(self, content):
        """חילוץ סיכום - עכשיו גם מנורמל"""
        # הסיכום ייצור מהתוכן שכבר נוקה אבל עוד לא נורמל
        # אז נחלץ סיכום רגיל ואז ננרמל אותו
        paragraphs = content.split('\n\n')
        summary_text = ""

        for paragraph in paragraphs:
            if len(paragraph.strip()) > 100:
                summary_text = paragraph.strip()[:500]
                break

        # נרמול הסיכום
        return self.normalize_text_for_training(summary_text)

    def extract_categories(self, wikicode):
        """חילוץ קטגוריות"""
        categories = []
        try:
            for link in wikicode.filter_wikilinks():
                title = str(link.title)
                if title.startswith('קטגוריה:'):
                    category = title.replace('קטגוריה:', '').strip()
                    if category:
                        categories.append(category)
        except Exception as e:
            print(f"Error extracting categories: {e}")
        return categories

    def extract_infobox(self, wikicode):
        """חילוץ אינפובוקס - גם מנורמל"""
        try:
            for template in wikicode.filter_templates():
                template_name = str(template.name).lower().strip()

                # דפוסים לזיהוי אינפובוקס
                infobox_patterns = [
                    'אינפו', 'infobox', 'תיבת מידע', 'מידע', 'תיבה',
                    'מנהיג', 'אדם', 'עיר', 'מדינה', 'ספר', 'סרט',
                    'חברה', 'ארגון', 'אוניברסיטה', 'בניין'
                ]

                if any(pattern in template_name for pattern in infobox_patterns):
                    # לא תבניות ניווט
                    exclude_patterns = ['ניווט', 'nav', 'citation', 'cite', 'ref', 'הערה']
                    if any(exclude in template_name for exclude in exclude_patterns):
                        continue

                    infobox_text = f"תיבת מידע - {template.name}: "  # שינוי: רווח במקום \n
                    param_count = 0

                    for param in template.params:
                        try:
                            param_name = str(param.name).strip()
                            param_value = str(param.value).strip()

                            if (param_name and param_value and
                                    len(param_value) < 300 and
                                    len(param_value) > 2 and
                                    not param_name.isdigit()):

                                # ניקוי בסיסי של הערך
                                param_value = re.sub(r'\[\[([^]|]*)\|?[^]]*\]\]', r'\1', param_value)
                                param_value = re.sub(r'\{\{[^}]*\}\}', '', param_value)
                                param_value = re.sub(r'<[^>]*>', '', param_value)
                                param_value = param_value.strip()

                                if param_value:
                                    infobox_text += f"{param_name}: {param_value} "  # רווח במקום \n
                                    param_count += 1

                                    if param_count >= 15:
                                        break

                        except Exception:
                            continue

                    if param_count > 0:
                        # נרמול האינפובוקס
                        return self.normalize_text_for_training(infobox_text)

        except Exception as e:
            print(f"Error extracting infobox: {e}")

        return ""

    def extract_tables(self, wikicode):
        """חילוץ טבלאות (פשוט)"""
        tables = []
        try:
            content = str(wikicode)
            table_matches = re.findall(r'\{\|.*?\|\}', content, re.DOTALL)
            for i, table in enumerate(table_matches[:3]):
                if len(table) < 1000:
                    tables.append(f"טבלה {i + 1}: [מידע מטבלה]")
        except Exception as e:
            print(f"Error extracting tables: {e}")
        return tables

    def calculate_quality(self, processed):
        """חישוב ציון איכות פשוט"""
        score = 30

        # אורך טקסט מעובד
        content_length = len(processed['content'])
        if content_length > 3000:
            score += 30
        elif content_length > 1500:
            score += 20
        elif content_length > 500:
            score += 10

        # קיום קטגוריות
        if len(processed['categories']) > 0:
            score += 15
        if len(processed['categories']) > 2:
            score += 5

        # קיום אינפובוקס
        if processed['infobox']:
            score += 15

        # איכות סיכום
        if len(processed['summary']) > 100:
            score += 5

        return min(100, score)

    def analyze_sample_quality(self):
        """ניתוח איכות המדגם"""
        if not self.articles:
            return

        print("\n📊 Sample Quality Analysis:")
        print("=" * 40)

        # סטטיסטיקות בסיסיות
        lengths = [a['processed_length'] for a in self.articles]
        qualities = [a['quality_score'] for a in self.articles]

        print(f"📝 Articles: {len(self.articles)}")
        print(f"📏 Avg length: {sum(lengths) / len(lengths):.0f} chars")
        print(f"⭐ Avg quality: {sum(qualities) / len(qualities):.1f}")

        # התפלגות איכות
        high_quality = len([q for q in qualities if q >= 80])
        med_quality = len([q for q in qualities if 60 <= q < 80])
        low_quality = len([q for q in qualities if q < 60])

        print(f"🟢 High quality (80+): {high_quality}")
        print(f"🟡 Medium quality (60-79): {med_quality}")
        print(f"🔴 Low quality (<60): {low_quality}")

        # בדיקת תוכן
        with_infobox = len([a for a in self.articles if a.get('infobox')])
        with_categories = len([a for a in self.articles if a.get('categories')])

        print(f"📋 With infobox: {with_infobox}")
        print(f"🏷️ With categories: {with_categories}")

        # דוגמה מנורמלת
        if self.articles:
            first_article = self.articles[0]
            print(f"\n📄 דוגמה מנורמלת (100 תווים ראשונים):")
            print(f"'{first_article['content'][:100]}...'")

    def save_sample(self):
        """שמירת הדגימה"""
        if not self.articles:
            print("❌ No articles to save!")
            return

        # יצירת מטא-דאטה
        lengths = [a['processed_length'] for a in self.articles]
        qualities = [a['quality_score'] for a in self.articles]

        metadata = {
            'total_articles': len(self.articles),
            'average_length': sum(lengths) / len(lengths),
            'average_quality': sum(qualities) / len(qualities),
            'total_categories': sum(len(a['categories']) for a in self.articles),
            'articles_with_infobox': sum(1 for a in self.articles if a['infobox']),
            'articles_with_categories': sum(1 for a in self.articles if a['categories']),
            'quality_distribution': {
                'high_80_plus': len([q for q in qualities if q >= 80]),
                'medium_60_79': len([q for q in qualities if 60 <= q < 80]),
                'low_below_60': len([q for q in qualities if q < 60])
            },
            'text_format': 'normalized_for_training'  # מידע על הפורמט
        }

        output_data = {
            'metadata': metadata,
            'articles': self.articles
        }

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n💾 Sample saved to {self.output_file}")
        print(f"📝 Total articles: {metadata['total_articles']}")
        print(f"📏 Average length: {metadata['average_length']:.0f} chars")
        print(f"⭐ Average quality: {metadata['average_quality']:.1f}")
        print(f"📋 Articles with infobox: {metadata['articles_with_infobox']}")
        print(f"🏷️ Articles with categories: {metadata['articles_with_categories']}")
        print(f"🔄 Text format: Normalized for training (no newlines)")

    def run_sample(self):
        """הרצת המדגם המלא"""
        print("🚀 Starting Wikipedia Hebrew sample processing...")
        print("🔄 Using normalized text format (no newlines) for training")
        print("=" * 60)

        # שלב 1: אוסף דפים תקינים
        print("🔍 Step 1: Collecting valid pages...")
        self.collect_valid_pages()

        if len(self.valid_pages) == 0:
            print("❌ No valid pages found!")
            return

        print(f"✅ Found {len(self.valid_pages)} valid pages")
        print()

        # שלב 2: יוצר דגימה ומעבד
        print("⚙️ Step 2: Creating sample and processing...")
        self.create_sample()

        if not self.articles:
            print("❌ No articles were successfully processed!")
            return

        print()

        # שלב 3: ניתוח איכות
        print("📊 Step 3: Analyzing quality...")
        self.analyze_sample_quality()
        print()

        # שלב 4: שומר תוצאות
        print("💾 Step 4: Saving results...")
        self.save_sample()

        print("\n🎉 Sample processing completed successfully!")
        print("📋 Text is normalized for training - no newlines or escape characters")


# שימוש
if __name__ == "__main__":
    processor = WikipediaSampleProcessor(
        dump_path=r'C:\Users\Dan Revital\OneDrive\Documents\gepeta\data\wikipedia\hewiki-latest-pages-articles.xml.bz2',
        output_file=r'C:\Users\Dan Revital\OneDrive\Documents\gepeta\load_wikipedia\hebrew_wiki_sample_100_normalized.json',
        sample_size=100
    )

    processor.run_sample()