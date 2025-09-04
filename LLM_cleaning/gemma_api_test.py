#!/usr/bin/env python3
"""
Gepeta Project - Google Gemma API Test Script
בדיקת ביצועים של Google API לניקוי טקסטים עבריים

Usage:
1. pip install google-generativeai boto3 pandas python-dotenv
2. הגדר API_KEY בקובץ .env או ישירות בקוד
3. python gepeta_api_test.py
"""

import google.generativeai as genai
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
import pandas as pd
from io import StringIO
from dotenv import load_dotenv

# =============================================================================
# הגדרות
# =============================================================================

# טען משתני סביבה
load_dotenv()

# הגדר את המפתח כאן או בקובץ .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# AWS S3 Settings (אם יש)
AWS_BUCKET = "gepeta-datasets"
AWS_PREFIX = "partly-processed/regex-and-dedup/"

# הגדרות עיבוד
MAX_WORKERS = 3  # מספר threads מקבילים
BATCH_SIZE = 50  # גודל באצ' לעיבוד
TEST_SIZE = 6  # מספר טקסטים לבדיקה


class GepetaAPITester:
    """מחלקה לבדיקת ביצועים של Google API"""

    def __init__(self, api_key):
        """אתחול עם מפתח API"""
        if api_key == "YOUR_API_KEY_HERE":
            raise ValueError("❌ עדכן את GOOGLE_API_KEY עם המפתח שלך!")

        genai.configure(api_key=api_key)
        self.model_name = 'gemma-3-27b-it'
        self.s3_client = None

        print("🔑 Google AI מוגדר לפרויקט Gepeta")
        print(f"⏰ זמן התחלה: {datetime.now().strftime('%H:%M:%S')}")

    def setup_s3(self):
        """הגדרת S3 client"""
        try:
            self.s3_client = boto3.client('s3')
            print("✅ S3 client מוכן")
            return True
        except Exception as e:
            print(f"⚠️ בעיה ב-S3: {e}")
            return False

    def test_single_request(self):
        """בדיקה בסיסית של בקשה יחידה"""
        print("\n🧪 בודק Google Gemma API...")

        try:
            model = genai.GenerativeModel(self.model_name)

            test_text = """דיווח: תא דאעש שנחשף בירדן תכנן לפגוע באנשי עסקים ישראליים 
© סופק על ידי מעריב תא דאעש... ____________________________________________________________ 
סרטונים שווים ב-MSN (BuzzVideos) אתר ע"י לינקטק ישראל"""

            prompt = f"""נקה את הטקסט העברי הבא מפגמי קידוד, תגיות HTML, פרסומות ותבניות. החזר רק טקסט נקי בעברית:

{test_text}

טקסט נקי:"""

            print("🚀 שולח בקשה ל-Google API...")
            start_time = time.time()

            response = model.generate_content(prompt)

            api_time = time.time() - start_time

            print(f"✅ תגובה התקבלה!")
            print(f"⚡ זמן API: {api_time:.2f} שניות")
            print(f"\n📝 תוצאה:")
            print(f"מקור: {test_text[:80]}...")
            print(f"נוקה: {response.text}")

            return True, api_time, response.text

        except Exception as e:
            print(f"❌ שגיאה ב-API: {e}")
            print("💡 בדוק:")
            print("1. שהמפתח נכון")
            print("2. שהפרויקט Gepeta פעיל")
            print("3. שיש גישה ל-Gemma API")
            return False, None, None

    def clean_single_text_api(self, text):
        """ניקוי טקסט יחיד עם Google API"""
        model = genai.GenerativeModel(self.model_name)

        prompt = f"""נקה את הטקסט העברי הבא מפגמי קידוד, תגיות HTML, פרסומות ותבניות. החזר רק טקסט נקי בעברית:

{text[:500]}

טקסט נקי:"""

        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"[API_ERROR] {str(e)}"

    def test_parallel_processing(self, texts, max_workers=3):
        """בדיקת עיבוד מקבילי"""
        print(f"\n⚡ בודק עיבוד מקבילי עם {max_workers} workers...")
        print(f"📝 מעבד {len(texts)} טקסטים...")

        start_time = time.time()
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # שליחת כל הבקשות
            future_to_text = {
                executor.submit(self.clean_single_text_api, text): i
                for i, text in enumerate(texts)
            }

            # איסוף תוצאות
            completed = 0
            for future in as_completed(future_to_text):
                result = future.result()
                results.append(result)
                completed += 1

                if completed % 2 == 0:  # עדכון כל 2 טקסטים
                    elapsed = time.time() - start_time
                    print(f"  ✅ הושלמו {completed}/{len(texts)} ({elapsed:.1f}s)")

        total_time = time.time() - start_time
        avg_time = total_time / len(texts)

        print(f"📊 תוצאות מקביליות:")
        print(f"  זמן כולל: {total_time:.2f}s")
        print(f"  זמן ממוצע לטקסט: {avg_time:.2f}s")
        print(f"  קצב: {len(texts) / total_time:.1f} texts/sec")

        return results, avg_time

    def run_performance_tests(self):
        """הרצת כל בדיקות הביצועים"""
        print("=" * 60)
        print("🎯 בדיקת ביצועים לפרויקט Gepeta")
        print("=" * 60)

        # בדיקה ראשונית
        api_works, single_time, result = self.test_single_request()

        if not api_works:
            print("\n❌ API לא עובד - בדוק את הבעיות למעלה")
            return False

        # השוואה מהירה למערכת המקומית
        local_avg_time = 2.5  # מהבדיקות הקודמות
        improvement = ((local_avg_time - single_time) / local_avg_time * 100)

        print(f"\n📊 השוואה מהירה:")
        print(f"🔴 מערכת מקומית: {local_avg_time:.2f}s")
        print(f"🟢 Google API: {single_time:.2f}s")
        print(f"📈 שיפור: {improvement:+.1f}%")

        # טקסטים לבדיקת מקביליות
        test_texts = [
            "דיווח: תא דאעש שנחשף בירדן תכנן לפגוע באנשי עסקים ישראליים © מעריב",
            "סוחר שהפיץ נפצים באשדוד וערים אחרות הופלל בוואטסאפ אלה רוזנבלט",
            "ההפגנות בירושלים נמשכות למרות הגשם שהתחזק בשעות הערב",
            "קבוצת הטלגרם החדשה למכירת דירות בתל אביב: אל תפספסו את ההזדמנות!",
            "ח\"כ דורון צור הגיב על האירועים הבטחוניים האחרונים בצפון הארץ",
            "חברת ההייטק הישראלית גייסה 50 מיליון דולר בסיבוב מימון חדש"
        ]

        # בדיקה עם workers שונים
        best_time = float('inf')
        best_workers = 1

        for workers in [10, 20]:
            print(f"\n🔬 בודק עם {workers} workers:")

            try:
                test_subset = test_texts[:min(workers + 1, len(test_texts))]
                results, avg_time = self.test_parallel_processing(test_subset, max_workers=workers)

                if avg_time < best_time:
                    best_time = avg_time
                    best_workers = workers

                # הערכה לכל הדאטא
                total_texts = 2454000
                estimated_days = (total_texts * avg_time) / 86400

                print(f"🔮 הערכה לכל הדאטא: {estimated_days:.1f} ימים")

                # בדיקת שגיאות API
                error_count = sum(1 for r in results if "API_ERROR" in str(r))
                if error_count > 0:
                    print(f"⚠️ שגיאות API: {error_count}/{len(results)} - אולי rate limit")
                    if error_count > len(results) / 2:
                        print("❌ יותר מדי שגיאות - עוצר בדיקה")
                        break

            except Exception as e:
                print(f"❌ שגיאה עם {workers} workers: {e}")
                break

        # סיכום ביצועים
        self.print_performance_summary(single_time, best_time, best_workers)

        return True

    def print_performance_summary(self, single_time, parallel_time, best_workers):
        """הדפסת סיכום ביצועים"""
        print(f"\n{'=' * 60}")
        print("🎯 סיכום ביצועים לפרויקט Gepeta")
        print("=" * 60)

        # נתונים
        total_texts = 2454000
        local_time = 2.5  # מערכת מקומית

        # חישובים
        local_days = (total_texts * local_time) / 86400
        api_single_days = (total_texts * single_time) / 86400
        api_parallel_days = (total_texts * parallel_time) / 86400

        print(f"📊 השוואת אפשרויות:")
        print(f"🔴 מערכת מקומית (SageMaker): {local_days:.1f} ימים")
        print(f"🟡 Google API (יחיד): {api_single_days:.1f} ימים")
        print(f"🟢 Google API (מקבילי): {api_parallel_days:.1f} ימים")
        print(f"🏆 הגדרה מיטבית: {best_workers} workers")

        # חישוב עלויות
        ai_studio_days = total_texts / 21600  # rate limit: 15/min = 21,600/day
        vertex_cost = total_texts * 0.0001  # הערכה גסה

        print(f"\n💰 אופציות עלות:")
        print(f"🆓 AI Studio (חינם): {ai_studio_days:.0f} ימים (rate limited)")
        print(f"💳 Vertex AI (בתשלום): {api_parallel_days:.1f} ימים (~${vertex_cost:,.0f})")

        # המלצה
        print(f"\n💡 המלצות לפרויקט Gepeta:")

        if api_parallel_days < 30:
            print("🎉 Google API מצוין! פחות מחודש!")
            recommendation = "עבור ל-Vertex AI לייצור מיידי"
        elif api_parallel_days < local_days / 2:
            print("✅ Google API משתלם!")
            if vertex_cost < 10000:  # פחות מ-10K דולר
                recommendation = "עבור ל-Vertex AI"
            else:
                recommendation = "שקול היברידי: חלק API, חלק מקומי"
        else:
            print("🤔 שיפור קטן בAPI")
            recommendation = "המשך עם המערכת המקומית + אופטימיזציות"

        print(f"🎯 המלצה: {recommendation}")


def main():
    """פונקציה ראשית"""
    print("🚀 Gepeta Project - Google API Performance Test")
    print("=" * 60)

    try:
        # יצירת tester
        tester = GepetaAPITester(GOOGLE_API_KEY)

        # הגדרת S3 (אופציונלי)
        tester.setup_s3()

        # הרצת בדיקות
        success = tester.run_performance_tests()

        if success:
            print(f"\n✅ בדיקה הושלמה בהצלחה!")
            print(f"⏰ זמן סיום: {datetime.now().strftime('%H:%M:%S')}")

            print(f"\n🚀 השלבים הבאים:")
            print("1. אם מרוצה מהביצועים → עבור ל-Vertex AI")
            print("2. אחרת → המשך עם אופטימיזציות מקומיות")
            print("3. שקול גישה היברידית")
        else:
            print(f"\n❌ בדיקה נכשלה - תקן את הבעיות למעלה")

    except ValueError as e:
        print(f"❌ {e}")
        print("💡 עדכן את GOOGLE_API_KEY בתחילת הקובץ")
    except Exception as e:
        print(f"❌ שגיאה כללית: {e}")

    print("\n🎯 פרויקט Gepeta - בדיקה הושלמה")


if __name__ == "__main__":
    main()