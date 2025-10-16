#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Wikipedia Text Cleaner Pipeline
======================================

ממשק פשוט לניקוי טקסט ויקיפדיה - מתודה אחת שמקבלת טקסט ומחזירה טקסט נקי.
"""

from wiki_text_cleaner import WikipediaTextCleaner


class SimpleWikipediaCleaner:
    """מנקה ויקיפדיה פשוט עם מתודה אחת"""

    def __init__(self):
        """אתחול המנקה"""
        self.cleaner = WikipediaTextCleaner()

    def clean_text(self, text: str) -> str:
        """
        מנקה טקסט ומחזיר טקסט נקי

        Args:
            text: טקסט גולמי

        Returns:
            טקסט נקי (מחרוזת ריקה אם הטקסט נפסל)
        """
        cleaned = self.cleaner.clean_text_only(text, min_length=0)

        if cleaned is None:
            cleaned = ""

        return cleaned


def main():
    """
    דוגמה לשימוש - זה בדיוק מה שאתה צריך לעשות!
    """
    print("🧹 ממשק ניקוי ויקיפדיה פשוט")
    print("=" * 50)

    # צור מנקה
    cleaner = SimpleWikipediaCleaner()

    # דוגמאות טקסט לניקוי
    test_texts = [
        # טקסט עם ויקי markup
        """{{תבנית|פרמטר=ערך}}
== כותרת ==
זה ערך עם [[קישור]] ו<ref>הפניה</ref>.

יש כאן גם   רווחים מרובים   ושורות\n\n\n\nרבות.""",

        # טקסט עם בעיות רווחים
        "זה טקסט   עם רווחים    מרובים ו\n\n\n\nשורות חדשות רבות",

        # טקסט עם PII
        "פנה אליי בכתובת: john@example.com או בכתובת IP: 192.168.1.100",

        # ערך הפניה
        "#REDIRECT [[ערך אחר]]",

        # טקסט רגיל
        "זה טקסט רגיל ללא בעיות מיוחדות."
    ]

    # נקה כל טקסט
    for i, text in enumerate(test_texts, 1):
        print(f"\n🔄 דוגמה {i}:")
        print(f"לפני: {repr(text)}")

        # זה הקוד הפשוט שאתה צריך!
        cleaned_text = cleaner.clean_text(text)

        print(f"אחרי: {repr(cleaned_text)}")
        print(f"תוצאה:\n{cleaned_text}")
        print("-" * 30)

    print("\n✅ זהו! זה כל מה שאתה צריך:")
    print("""
# יצירת מנקה
cleaner = SimpleWikipediaCleaner()

# ניקוי טקסט
clean_text = cleaner.clean_text(your_text)

# זהו!
""")


if __name__ == "__main__":
    main()