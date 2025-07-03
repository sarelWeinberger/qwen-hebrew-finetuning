#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikipedia Article Finder
=========================

חיפוש ערך ספציפי בדאמפ ויקיפדיה, ניקוי, ושמירה לקובץ.
משתמש במודול הניקוי המרכזי.
"""

import xml.etree.ElementTree as ET
import bz2
from pathlib import Path
from typing import Optional

from wiki_text_cleaner import WikipediaTextCleaner, count_words, count_bytes


class WikipediaArticleFinder:
    """מחלקה לחיפוש וניקוי ערך ספציפי"""

    def __init__(self, dump_path: str, save_examples: bool = False):
        """
        אתחול החיפוש

        Args:
            dump_path: נתיב לקובץ הדאמפ
            save_examples: האם לשמור דוגמאות ניקוי (ברירת מחדל: False)
        """
        self.dump_path = dump_path
        self.save_examples = save_examples

        # יצירת מנקה
        if save_examples:
            self.cleaner = WikipediaTextCleaner(example_callback=self._save_example_to_file)
            self.examples_saved = {}
        else:
            self.cleaner = WikipediaTextCleaner()

    def _save_example_to_file(self, category: str, raw_text: str, clean_text: str):
        """שמירת דוגמאות לקבצים מקומיים"""
        if category not in self.examples_saved:
            self.examples_saved[category] = 0

        # הגבלה ל-10 דוגמאות לכל קטגוריה
        if self.examples_saved[category] >= 10:
            return

        examples_dir = Path("examples")
        examples_dir.mkdir(exist_ok=True)

        example_file = examples_dir / f"{category}_examples.txt"

        with open(example_file, 'a', encoding='utf-8') as f:
            f.write(f"=== דוגמה {self.examples_saved[category] + 1} ===\n")
            f.write("BEFORE:\n")
            f.write(raw_text[:500] + "...\n" if len(raw_text) > 500 else raw_text + "\n")
            f.write("\nAFTER:\n")
            f.write(clean_text[:500] + "...\n" if len(clean_text) > 500 else clean_text + "\n")
            f.write("\n" + "=" * 50 + "\n\n")

        self.examples_saved[category] += 1
        print(f"📝 נשמרה דוגמה {category} ({self.examples_saved[category]}/10)")

    def find_article_in_dump(self, article_title: str) -> Optional[str]:
        """
        מוצא ערך ספציפי בדאמפ ויקיפדיא

        Args:
            article_title: שם הערך לחיפוש

        Returns:
            ויקי-טקסט גולמי או None אם לא נמצא
        """
        print(f"🔍 מחפש ערך: '{article_title}' בדאמפ...")
        print(f"📂 דאמפ: {self.dump_path}")
        print("⏳ זה עלול לקחת זמן...")

        scanned_pages = 0

        try:
            with bz2.open(self.dump_path, 'rt', encoding='utf-8') as file:
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

        except Exception as e:
            print(f"❌ שגיאה בחיפוש: {e}")
            return None

        print(f"❌ לא נמצא ערך: '{article_title}' (סרקתי {scanned_pages:,} דפים)")
        return None

    def process_article(self, article_title: str, output_dir: str = ".") -> bool:
        """
        מחפש, מנקה ושומר ערך

        Args:
            article_title: שם הערך
            output_dir: תיקיית פלט

        Returns:
            True אם הצליח, False אחרת
        """
        print(f"🎯 מעבד ערך: '{article_title}'")
        print("=" * 60)

        # שלב 1: חיפוש הערך
        raw_wikitext = self.find_article_in_dump(article_title)

        if not raw_wikitext:
            print(f"❌ לא נמצא הערך '{article_title}'")
            return False

        print()

        # שלב 2: ניקוי הערך
        print("⚙️ מנקה את הויקי-טקסט...")

        cleaned_text = self.cleaner.clean_article(article_title, raw_wikitext)

        if not cleaned_text:
            print("❌ הערך נפסל בעיבוד (ייתכן שהוא הפניה או קצר מדי)")
            return False

        print(f"✅ ניקוי הושלם בהצלחה")
        print(f"📄 אורך אחרי עיבוד: {len(cleaned_text):,} תווים")

        # שלב 3: שמירת התוצאות
        print("=" * 60)
        print("💾 שומר תוצאות...")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # שמירת הטקסט הגולמי
        safe_title = article_title.replace(' ', '_').replace('/', '_')
        raw_filename = output_path / f"{safe_title}_raw_wikitext.txt"
        with open(raw_filename, 'w', encoding='utf-8') as f:
            f.write(raw_wikitext)
        print(f"✅ ויקי-טקסט גולמי נשמר: {raw_filename}")

        # שמירת הטקסט המעובד
        processed_filename = output_path / f"{safe_title}_cleaned.txt"
        with open(processed_filename, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        print(f"✅ טקסט מעובד נשמר: {processed_filename}")

        # שמירת מטאדטה
        metadata_filename = output_path / f"{safe_title}_metadata.txt"
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            f.write(f"כותרת: {article_title}\n")
            f.write(f"אורך גולמי: {len(raw_wikitext):,} תווים\n")
            f.write(f"אורך נקי: {len(cleaned_text):,} תווים\n")
            f.write(f"מילים: {count_words(cleaned_text):,}\n")
            f.write(f"בייטים: {count_bytes(cleaned_text):,}\n")
            f.write(f"דחיסה: {(1 - len(cleaned_text) / len(raw_wikitext)) * 100:.1f}%\n")

            # סטטיסטיקות ניקוי
            cleaning_stats = self.cleaner.get_stats()
            if cleaning_stats:
                f.write(f"\nסטטיסטיקות ניקוי:\n")
                for stat_name, count in cleaning_stats.items():
                    if count > 0:
                        f.write(f"  {stat_name}: {count}\n")

        print(f"✅ מטאדטה נשמרה: {metadata_filename}")

        # סיכום
        print("\n" + "=" * 60)
        print("🎉 הושלם בהצלחה!")
        print(f"📊 סיכום:")
        print(f"   ויקי-טקסט גולמי: {len(raw_wikitext):,} תווים")
        print(f"   טקסט מעובד: {len(cleaned_text):,} תווים")
        print(f"   מילים: {count_words(cleaned_text):,}")
        print(f"   דחיסה: {(1 - len(cleaned_text) / len(raw_wikitext)) * 100:.1f}%")

        if self.save_examples and hasattr(self, 'examples_saved'):
            total_examples = sum(self.examples_saved.values())
            if total_examples > 0:
                print(f"   דוגמאות שנשמרו: {total_examples}")

        print(f"\n📄 דוגמה מהטקסט המעובד (200 תווים ראשונים):")
        print(f"   {cleaned_text[:200]}...")

        return True


def main():
    """הפעלה ראשית"""
    print("🎯 חיפוש וניקוי ערך ויקיפדיה - מערכת מאוחדת")
    print("=" * 60)

    # הגדרות
    dump_path = r'C:\Users\Dan Revital\OneDrive\Documents\gepeta\data\wikipedia\hewiki-latest-pages-articles.xml.bz2'
    article_title = 'קוד פתוח'
    output_dir = "article_output"
    save_examples = True  # שמירת דוגמאות ניקוי

    print(f"🔍 מחפש ערך: '{article_title}'")
    print(f"📂 בדאמפ: {dump_path}")
    print(f"📁 פלט: {output_dir}")
    print(f"📝 שמירת דוגמאות: {'כן' if save_examples else 'לא'}")
    print()

    # יצירת חיפוש
    finder = WikipediaArticleFinder(dump_path, save_examples=save_examples)

    # עיבוד הערך
    success = finder.process_article(article_title, output_dir)

    if not success:
        print("❌ העיבוד נכשל")
        return

    print("\n🎉 התהליך הושלם בהצלחה!")


if __name__ == "__main__":
    main()