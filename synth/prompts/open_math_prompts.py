GSM_INSTRUCT_GEMINI_V1 = """Your task is to translate the given English question and answer into Hebrew. Follow these guidelines:

1. Translate only the question and answer options provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the question and answers as accurately as possible in the Hebrew translation.
3. Maintain the same formatting as the original English version.
4. Write the translations in a style suitable for grade school-level math questions."""

GSM_SYNTH_INSTRUCT_GEMINI_V1 = """
Your task is to translate the given English question and answer into Hebrew, describe your reasoning in Hebrew as well. You will be given the final answer which is either a number or a LaTex equation. Follow these guidelines:

1. Translate only the question and answer options provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the question and answers as accurately as possible in the Hebrew translation.
3. Maintain the same formatting as the original English version.
"""

#### Based on GSM
GSM_ENGLISH_QUESTIONS = ["""
In April, Tank gathered 10 more Easter eggs than Emma in their first round of egg hunt. However, Emma gathered twice as many eggs as Tank in their second round of egg hunt, while Tank's total number of eggs in the second round was 20 less than the number she had gathered in the first round. If the total number of eggs in the pile they were collecting with 6 other people was 400 eggs, and Emma gathered 60 eggs in the second egg hunt round, find the number of eggs that the 6 other egg hunters collected?
""",
"Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
,"Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
"Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?"
, "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?"]

GSM_ENGLISH_ANSWERS = ["""
                       Tank's total number of eggs in the second round was 60/2=<<60/2=30>>30 since Emma gathered twice as many eggs as Tank in their second round of egg hunt.
The total number of eggs that Emma and Tank gathered in the second round was 60+30=<<60+30=90>>90
Tank's total number of eggs in the second round was 20 less than the number she had gathered in the first round, meaning she had gathered 30+20=<<30+20=50>>50 eggs in the first round of egg hunt.
Tank gathered 10 more Easter eggs than Emma in their first round of egg hunt, meaning Emma collected 50-10=40 eggs
The total number of eggs Emma and Tank collected in the first round was 40+50=<<40+50=90>>90
In the two rounds, Emma and Tank gathered 90+90=<<90+90=180>>180 eggs
If the total number of eggs in the pile they were collecting with 6 other people was 400 eggs, the six other people gathered 400-180=<<400-180=220>>220 eggs
#### 220
                       """,
"""
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10""", """
In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.
Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.
This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.
#### 5
""",
"""
Maila read 12 x 2 = <<12*2=24>>24 pages today.
So she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday.
There are 120 - 36 = <<120-36=84>>84 pages left to be read.
Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = <<84/2=42>>42 pages.
#### 42
""",
"""
He writes each friend 3*2=<<3*2=6>>6 pages a week
So he writes 6*2=<<6*2=12>>12 pages every week
That means he writes 12*52=<<12*52=624>>624 pages a year
#### 624
"""
]

GSM_HEBREW_QUESTIONS= ["באפריל, טאנק אספה 10 ביצי פסחא יותר מאמה בסבב הראשון של ציד ביצים. עם זאת, אמה אספה פי שניים ביצים מטאנק בסבב השני של ציד ביצים, בעוד שהמספר הכולל של הביצים של טאנק בסבב השני היה 20 פחות מהמספר שהיא אספה בסבב הראשון. אם המספר הכולל של הביצים בערימה שהן אספו יחד עם 6 אנשים נוספים היה 400 ביצים, ואמה אספה 60 ביצים בסבב השני של ציד ביצים, מצאו את מספר הביצים שאספו 6 ציידי הביצים האחרים?",
                       "וונג מרוויחה 12$ לשעה עבור שמרטפות. אתמול, היא עבדה רק 50 דקות כשמרטפית. כמה היא הרוויחה?",
                       "בטי חוסכת כסף לארנק חדש שעולה 100$. לבטי יש רק חצי מהכסף שהיא צריכה. הוריה החליטו לתת לה 15$ למטרה זו, וסבא וסבתא שלה נתנו לה פי שניים ממה שהוריה נתנו. כמה כסף נוסף בטי צריכה כדי לקנות את הארנק?",
                       "ג'ולי קוראת ספר בן 120 עמודים. אתמול, היא הצליחה לקרוא 12 עמודים והיום, היא קראה פי שניים עמודים מאשר אתמול. אם היא רוצה לקרוא מחר מחצית מהעמודים הנותרים, כמה עמודים עליה לקרוא?",
                       "ג'יימס כותב מכתב בן 3 עמודים לשני חברים שונים פעמיים בשבוע. כמה עמודים הוא כותב בשנה?"]

GSM_HEBREW_ANSWERS = ["""מספר ביצי הפסחא הכולל של טאנק בסיבוב השני היה 60/2=<<60/2=30>>30 מכיוון שאמה אספה פי שניים ביצי פסחא מטנק בסיבוב השני שלהן בציד הביצים.
מספר ביצי הפסחא הכולל שאמה וטאנק אספו בסיבוב השני היה 60+30=<<60+30=90>>90
מספר ביצי הפסחא הכולל של טאנק בסיבוב השני היה 20 פחות ממספר הביצים שהיא אספה בסיבוב הראשון, כלומר היא אספה 30+20=<<30+20=50>>50 ביצי פסחא בסיבוב הראשון של ציד הביצים.
טאנק אספה 10 ביצי פסחא יותר מאמה בסיבוב הראשון שלהן בציד הביצים, כלומר אמה אספה 50-10=40 ביצי פסחא
מספר ביצי הפסחא הכולל שאמה וטאנק אספו בסיבוב הראשון היה 40+50=<<40+50=90>>90
בשני הסיבובים, אמה וטאנק אספו 90+90=<<90+90=180>>180 ביצי פסחא
אם המספר הכולל של ביצי הפסחא בערימה שהן אספו יחד עם 6 אנשים אחרים היה 400 ביצים, ששת האנשים האחרים אספו 400-180=<<400-180=220>>220 ביצי פסחא
#### 220""",
"""וונג מרוויחה 12/60 = $<<12/60=0.2>>0.2 לדקה.
בעבודה של 50 דקות, היא הרוויחה 0.2 × 50 = $<<0.2*50=10>>10.
#### 10""",
"""בהתחלה, לבטי יש רק 100 / 2 = $<<100/2=50>>50.
סבא וסבתא של בטי נתנו לה 15 * 2 = $<<15*2=30>>30.
זה אומר שבטי צריכה עוד 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5.
#### 5""",
"""ג'ולי קראה 12 × 2 = <<12*2=24>>24 עמודים היום.
אז היא הצליחה לקרוא סך הכל 12 + 24 = <<12+24=36>>36 עמודים מאתמול.
נותרו 120 - 36 = <<120-36=84>>84 עמודים לקריאה.
מכיוון שהיא רוצה לקרוא מחר מחצית מהעמודים הנותרים, אז עליה לקרוא 84/2 = <<84/2=42>>42 עמודים.
#### 42""",
"""הוא כותב לכל חבר 3*2=<<3*2=6>>6 עמודים בשבוע
אז הוא כותב 6*2=<<6*2=12>>12 עמודים בכל שבוע
זה אומר שהוא כותב 12*52=<<12*52=624>>624 עמודים בשנה
#### 624"""]