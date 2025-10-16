#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json


def analyze_quotes_reality(json_file_path):
    """
    בדיקה האם \" זה רק הצגה של JSON או תווים אמיתיים
    """
    print("🔬 בדיקת המציאות של \" - הצגה VS תווים אמיתיים")
    print("=" * 60)

    try:
        # טעינת הקובץ
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"📂 נטען קובץ: {json_file_path}")
        print(f"📊 מספר ערכים: {data['metadata']['total_articles']}")

        # בחירת ערך לבדיקה
        if data['articles']:
            article = data['articles'][0]
            content = article['content']

            print(f"\n📄 בודק ערך: {article['title']}")
            print(f"📏 אורך התוכן: {len(content)} תווים")

            # בדיקות ספציפיות
            print("\n🔍 בדיקות מפורטות:")

            # 1. ספירת תווים
            backslash_count = content.count('\\')
            quote_count = content.count('"')
            backslash_quote_count = content.count('\\"')

            print(f"   \\ (backslash בלבד): {backslash_count}")
            print(f"   \" (מרכאות בלבד): {quote_count}")
            print(f"   \\\" (backslash + מרכאות): {backslash_quote_count}")

            # 2. בדיקת כל תו בנפרד
            print(f"\n🔬 בדיקת תווים בודדים (50 תווים ראשונים):")
            for i, char in enumerate(content[:50]):
                if char in ['"', '\\']:
                    ascii_code = ord(char)
                    print(f"   מיקום {i:2d}: '{char}' (ASCII: {ascii_code})")

            # 3. חיפוש מקרי מרכאות
            print(f"\n📍 מיקומי מרכאות ראשונים:")
            quote_positions = []
            for i, char in enumerate(content):
                if char == '"':
                    quote_positions.append(i)
                    if len(quote_positions) >= 3:  # מספיק 3 ראשונים
                        break

            for pos in quote_positions:
                start = max(0, pos - 10)
                end = min(len(content), pos + 10)
                context = content[start:end]
                print(f"   מיקום {pos}: '...{context}...'")

                # בדיקה אם לפני המרכאות יש backslash
                if pos > 0 and content[pos - 1] == '\\':
                    print(f"      ⚠️  יש backslash לפני המרכאות!")
                else:
                    print(f"      ✅ אין backslash לפני המרכאות")

            # 4. בדיקת אורך בפועל
            print(f"\n📊 בדיקת אורך:")

            # יצירת מחרוזת test
            test_with_backslash = 'המילה \\"test\\" במרכאות'
            test_without_backslash = 'המילה "test" במרכאות'

            print(f"   מחרוזת עם \\\":     '{test_with_backslash}' (אורך: {len(test_with_backslash)})")
            print(f"   מחרוזת עם \" בלבד: '{test_without_backslash}' (אורך: {len(test_without_backslash)})")

            # השוואה
            if '\\' in content:
                print(f"\n⚠️  נמצאו תווי backslash בתוכן!")
                print(f"     זה אומר שה-\\\" הם תווים אמיתיים, לא רק הצגה")
            else:
                print(f"\n✅ לא נמצאו תווי backslash בתוכן")
                print(f"     זה אומר שה-\\\" שאתה רואה זה רק הצגה של JSON")

            # 5. שמירה לקובץ טקסט רגיל לבדיקה
            test_file = "content_test.txt"
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(content[:500])  # 500 תווים ראשונים

            print(f"\n💾 נשמרו 500 תווים ראשונים לקובץ: {test_file}")
            print(f"     פתח את הקובץ בעורך טקסט רגיל ובדוק איך נראות המרכאות")

            # 6. הדפסה ישירה לקונסול
            print(f"\n🖥️  הדפסה ישירה (50 תווים ראשונים):")
            print(f"     {content[:50]}")

        else:
            print("❌ אין ערכים בקובץ")

    except FileNotFoundError:
        print(f"❌ קובץ לא נמצא: {json_file_path}")
    except Exception as e:
        print(f"❌ שגיאה: {e}")


def main():
    """
    הפעלה ראשית
    """
    print("🎯 מטרה: לבדוק האם \\\" זה רק הצגה של JSON או תווים אמיתיים")
    print()

    # בדיקת הקובץ
    json_file = "hebrew_wiki_sample_100_normalized.json"
    analyze_quotes_reality(json_file)

    print("\n" + "=" * 60)
    print("🏁 סיכום:")
    print("1. אם יש backslash בתוכן → זה תווים אמיתיים, צריך לתקן")
    print("2. אם אין backslash בתוכן → זה רק הצגה של JSON, הכל בסדר")
    print("3. בדוק גם את הקובץ content_test.txt שנוצר")


if __name__ == "__main__":
    main()