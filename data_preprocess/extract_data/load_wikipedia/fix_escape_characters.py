#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def fix_escape_characters(text):
    """
    מתקן תווי בריחה בטקסט
    \n -> שורה חדשה אמיתית
    \" -> מרכאות רגילות "
    """
    print("🔍 טקסט לפני תיקון:")
    print("=" * 50)
    print(repr(text[:200]) + "...")  # מציג את התווים הגולמיים
    print()

    # תיקון תווי בריחה
    fixed_text = text.replace('\\n', '\n')  # החלפת \n בשורה חדשה
    fixed_text = fixed_text.replace('\\"', '"')  # החלפת \" במרכאות רגילות
    fixed_text = fixed_text.replace('\\\\', '\\')  # החלפת \\\\ ב-\ יחיד
    fixed_text = fixed_text.replace('\\t', '\t')  # החלפת \t בטאב
    fixed_text = fixed_text.replace('\\r', '\r')  # החלפת \r בחזרת עגלה

    print("✅ טקסט אחרי תיקון:")
    print("=" * 50)
    print(fixed_text[:500] + "...")  # מציג את הטקסט המתוקן
    print()

    print("📊 סטטיסטיקות:")
    # ספירת תווי בריחה - צריך להגדיר משתנים בנפרד כי f-string לא מקבל backslash
    newline_count = text.count('\\n')
    quote_count = text.count('\\"')
    print(f"   \\n שהוחלפו: {newline_count}")
    print(f"   \\\" שהוחלפו: {quote_count}")
    print(f"   אורך לפני: {len(text)} תווים")
    print(f"   אורך אחרי: {len(fixed_text)} תווים")

    return fixed_text


def main():
    """
    קוראת קובץ wiki_test.txt ומתקנת תווי בריחה
    """
    input_file = "wiki_test.txt"
    output_file = "wiki_test_fixed.txt"

    try:
        # קריאת הקובץ
        print(f"📖 קורא קובץ: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"✅ נקרא קובץ בגודל {len(content)} תווים")
        print()

        # תיקון תווי בריחה
        fixed_content = fix_escape_characters(content)

        # שמירת הקובץ המתוקן
        print(f"💾 שומר קובץ מתוקן: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)

        print(f"✅ הקובץ נשמר בהצלחה!")

        # בדיקה נוספת - ספירת שורות
        lines_before = content.count('\n') + content.count('\\n')
        lines_after = fixed_content.count('\n')
        print(f"📝 שורות לפני: ~{lines_before}, אחרי: {lines_after}")

    except FileNotFoundError:
        print(f"❌ שגיאה: לא נמצא קובץ {input_file}")
        print("💡 ודא שהקובץ קיים באותה תיקייה")
    except Exception as e:
        print(f"❌ שגיאה: {e}")


if __name__ == "__main__":
    main()