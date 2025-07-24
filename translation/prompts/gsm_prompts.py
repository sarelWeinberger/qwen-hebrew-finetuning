"""
This file contains variables with prompts to be used when translating the 'GSM8K' benchmark:
https://huggingface.co/datasets/openai/gsm8k
"""

# Instruction prompts
GSM_INSTRUCT_CLAUDE_V1 = """<instruction>
Your task is to translate the given English question and answer into Hebrew. Follow these guidelines:

1. Translate only the question and answer options provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the question and answers as accurately as possible in the Hebrew translation.
3. Maintain the same formatting as the original English version.
4. Write the translations in a style suitable for grade school-level math questions.
</instruction>"""

GSM_INSTRUCT_CLAUDE_REFINE_V1 = """<instruction>
Your task is to translate the given English question and possible answers into Hebrew. First write a possible translation, then improve it. 
Follow these guidelines for the final translation:

1. Preserve the original semantic meaning and intent of the question and answers as accurately as possible in the Hebrew translation.
2. Maintain the same formatting as the original English version.
3. Write the translations in a style suitable for grade school-level math questions.
</instruction>"""

GSM_INSTRUCT_CLAUDE_REFINE_V2 = """<instruction>
Your task is to translate the given English question and possible answers into Hebrew. First write a possible translation, then explain how to improve it and write the final translation.
Follow these guidelines for the final translation:

1. Preserve the original semantic meaning and intent of the question and answers as accurately as possible in the Hebrew translation.
2. Maintain the same formatting as the original English version, keep the same numbers as the original.
3. Write the translations in a natural Hebrew style suitable for grade school-level math questions.
4. Adjust entities from American context into Israeli context, including names, currency, measuremnt units and etc. For example: map "USD" to "שקלים". Do not convert the numbers, only change the entities.
5. Adjust American scenarios into Israeli scenarios to make it more natural.
6. Cultural adaptation for rare activities: If the activity or context involves something very rare or unfamiliar in Israeli culture (like lacrosse, specific American transit systems, or highly location-specific activities), perform transcreation instead of direct translation - replace the acticity with a similar Israeli equivalent that maintains the same logical structure.
</instruction>"""

GSM_INSTRUCT_CLAUDE_MULTI_V1 = """
"""

GSM_INSTRUCT_GEMINI_V1 = """Your task is to translate the given English question and answer into Hebrew. Follow these guidelines:

1. Translate only the question and answer options provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the question and answers as accurately as possible in the Hebrew translation.
3. Maintain the same formatting as the original English version.
4. Write the translations in a style suitable for grade school-level math questions."""

GSM_INSTRUCT_GEMINI_V2 = """Your task is to translate the given English question and answer into Hebrew. Follow these guidelines:

1. Translate only the question and answer options provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the question and answers as accurately as possible in the Hebrew translation.
3. Maintain the same formatting as the original English version.
4. Write the translations in a natural Hebrew style suitable for grade school-level math questions.
5. Adjust entities from American context into Israeli context, including names, currency, measuremnt units and etc. For example: map "USD" to "שקלים".
6. Adjust American scenarios into Israeli scenarios to make it more natural.
7. Cultural adaptation for rare activities: If the activity or context involves something very rare or unfamiliar in Israeli culture (like lacrosse, specific American transit systems, or highly location-specific activities), perform transcreation instead of direct translation - replace the acticity with a similar Israeli equivalent that maintains the same logical structure."""

GSM_INSTRUCT_GEMINI_MULTI_V1 = """
"""

# Format prompts (to be used in claude's prompt)
GSM_FORMAT = """<response_format>
<question>Translated question</question>
<answer>Translated answer</answer>
</response_format>"""

GSM_FORMAT_REFINE = """<response_format>
First translation attempt:
<question>First attempt translated question</question>
<answer>First attempt translated answer</answer>

<explain>Explanation text.</explain>

Improved translation:
<question>Final translated question</question>
<answer>Final translated answer</answer>
</response_format>"""

# The few-shots samples
GSM_FEW_SHOTS = """<fewshot_examples>
<example>
English:
<question>In April, Tank gathered 10 more Easter eggs than Emma in their first round of egg hunt. However, Emma gathered twice as many eggs as Tank in their second round of egg hunt, while Tank's total number of eggs in the second round was 20 less than the number she had gathered in the first round. If the total number of eggs in the pile they were collecting with 6 other people was 400 eggs, and Emma gathered 60 eggs in the second egg hunt round, find the number of eggs that the 6 other egg hunters collected?</question>
<answer>Tank's total number of eggs in the second round was 60/2=<<60/2=30>>30 since Emma gathered twice as many eggs as Tank in their second round of egg hunt.
The total number of eggs that Emma and Tank gathered in the second round was 60+30=<<60+30=90>>90
Tank's total number of eggs in the second round was 20 less than the number she had gathered in the first round, meaning she had gathered 30+20=<<30+20=50>>50 eggs in the first round of egg hunt.
Tank gathered 10 more Easter eggs than Emma in their first round of egg hunt, meaning Emma collected 50-10=40 eggs
The total number of eggs Emma and Tank collected in the first round was 40+50=<<40+50=90>>90
In the two rounds, Emma and Tank gathered 90+90=<<90+90=180>>180 eggs
If the total number of eggs in the pile they were collecting with 6 other people was 400 eggs, the six other people gathered 400-180=<<400-180=220>>220 eggs
#### 220</answer>

Hebrew:
<question>באפריל, טניה אספה 10 ביצי פסחא יותר מאמה בסבב ציד הביצים הראשון שלהם. עם זאת, אמה אספה פי שניים ביצים מטניה בסבב השני שלהן של ציד הביצים, בעוד שהמספר הכולל של הביצים של טניה בסבב השני היה 20 פחות מהמספר שהיא אספה בסבב הראשון. אם סך כל הביצים בערימה שהן אספו יחד עם 6 אנשים נוספים היה 400 ביצים, ואמה אספה 60 ביצים בסבב ציד הביצים השני, מצאו את מספר הביצים שאספו 6 ציידי הביצים האחרים?</question>
<answer>מספר ביצי הפסחא הכולל של טניה בסיבוב השני היה 60/2=<<60/2=30>>30 מכיוון שאמה אספה פי שניים ביצים יותר מטניה בסיבוב השני שלהן בציד הביצים.
סך כל ביצי הפסחא שאמה וטניה אספו בסיבוב השני היה 60+30=<<60+30=90>>90
סך כל ביצי הפסחא של טניה בסיבוב השני היה 20 פחות מהמספר שהיא אספה בסיבוב הראשון, כלומר היא אספה 30+20=<<30+20=50>>50 ביצים בסיבוב הראשון של ציד הביצים.
טניה אספה 10 ביצי פסחא יותר מאמה בסיבוב הראשון שלהן בציד הביצים, כלומר אמה אספה 50-10=40 ביצי פסחא
סך כל ביצי הפסחא שאמה וטניה אספו בסיבוב הראשון היה 40+50=<<40+50=90>>90
בשני הסבבים, אמה וטניה אספו 90+90=<<90+90=180>>180 ביצי פסחא
אם המספר הכולל של ביצי הפסחא בערימה שהן אספו יחד עם 6 אנשים אחרים היה 400 ביצים, ששת האנשים האחרים אספו 400-180=<<400-180=220>>220 ביצים
#### 220</answer>
</example>

<example>
English:
<question>Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?</question>
<answer>Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10</answer>

Hebrew:
<question>נוגה מרוויחה 12₪ לשעה כבייביסיטרית. אתמול, היא עבדה רק 50 דקות כבייביסיטרית. כמה היא הרוויחה?</question>
<answer>נוגה מרוויחה 12/60 = ₪<<12/60=0.2>>0.2 לדקה.
בעבודה של 50 דקות, היא הרוויחה 0.2 × 50 = ₪<<0.2*50=10>>10.
#### 10</answer>
</example>

<example>
English:
<question>Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?</question>
<answer>In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.
Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.
This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.
#### 5</answer>

Hebrew:
<question>בטי חוסכת כסף לארנק חדש שעולה 100₪. לבטי יש רק חצי מהכסף שהיא צריכה. הוריה החליטו לתת לה 15₪ למטרה זו, וסבא וסבתא שלה, פי שניים ממה שהוריה נתנו. כמה עוד כסף בטי צריכה כדי לקנות את הארנק?</question>
<answer>בהתחלה, יש לבטי רק 100 / 2 = ₪<<100/2=50>>50.
סבא וסבתא של בטי נתנו לה 15 * 2 = ₪<<15*2=30>>30.
זה אומר שבטי צריכה עוד 100 - 50 - 30 - 15 = ₪<<100-50-30-15=5>>5.
#### 5</answer>
</example>

<example>
English:
<question>Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?</question>
<answer>Julie read 12 x 2 = <<12*2=24>>24 pages today.
So she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday.
There are 120 - 36 = <<120-36=84>>84 pages left to be read.
Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = <<84/2=42>>42 pages.
#### 42</answer>

Hebrew:
<question>ג'ולי קוראת ספר בן 120 עמודים. אתמול, היא הצליחה לקרוא 12 עמודים, והיום, היא קראה פי שניים עמודים מאשר אתמול. אם היא רוצה לקרוא מחר מחצית מהעמודים הנותרים, כמה עמודים עליה לקרוא?</question>
<answer>ג’ולי קראה 12 × 2 = <<12*2=24>>24 עמודים היום.
אז היא הצליחה לקרוא סך הכל 12 + 24 = <<12+24=36>>36 עמודים מאתמול.
יש 120 - 36 = <<120-36=84>>84 עמודים שנותרו לקרוא.
מכיוון שהיא רוצה לקרוא מחר מחצית מהעמודים הנותרים, אז היא צריכה לקרוא 84/2 = <<84/2=42>>42 עמודים.
#### 42</answer>
</example>

<example>
English:
<question>James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?</question>
<answer>He writes each friend 3*2=<<3*2=6>>6 pages a week
So he writes 6*2=<<6*2=12>>12 pages every week
That means he writes 12*52=<<12*52=624>>624 pages a year
#### 624</answer>

Hebrew:
<question>ג'יימס כותב מכתב בן 3 עמודים לשני חברים שונים פעמיים בשבוע. כמה עמודים הוא כותב בשנה?</question>
<answer>הוא כותב לכל חבר 3*2=<<3*2=6>>6 עמודים בשבוע
אז הוא כותב 6*2=<<6*2=12>>12 עמודים בכל שבוע
זה אומר שהוא כותב 12*52=<<12*52=624>>624 עמודים בשנה
#### 624</answer>
</example>
</fewshot_examples>"""
