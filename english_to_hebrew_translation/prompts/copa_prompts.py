"""
This file contains variables with prompts to be used when translating the 'Copa' benchmark:
https://huggingface.co/datasets/pkavumba/balanced-copa

This file contains only the chosen prompts to be used.
The file 'old_arc_prompts.py' contains old version prompts that we have tested.
"""

COPA_INSTRUCT_V1_GEMINI = """Your task is to translate the given English premise and possible choices into Hebrew. Follow these guidelines:

1. Translate only the premise and choices options provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the premise and choices as accurately as possible in the Hebrew english_to_hebrew_translation.
3. Maintain the same formatting as the original English version.
4. Write the translations in a clear and natural Hebrew style.
5. Pay special attention to maintaining the logical flow and coherence of the context when translating."""


# The few-shots samples
COPA_FEW_SHOTS_CAUSE = """<fewshot_examples>
<example>
English:
<premise>My body cast a shadow over the grass.</premise>
<choice1>The grass was cut.</choice1>
<choice2>The sun was rising.</choice2>
Hebrew:
<premise>הגוף שלי הטיל צל על הדשא.</premise>
<choice1>הדשא היה מכוסח.</choice1>
<choice2>השמש זרחה.</choice2>
</example>

<example>
English:
<premise>The woman tolerated her friend's difficult behavior.</premise>
<choice1>The woman knew her friend was going through a hard time.</choice1>
<choice2>The woman felt that her friend took advantage of her kindness.</choice2>
Hebrew:
<premise>האישה סבלה את ההתנהגות הקשה של חברתה.</premise>
<choice1>האישה ידעה שחברתה עוברת תקופה קשה.</choice1>
<choice2>האישה הרגישה שחברתה ניצלה את טוב ליבה.</choice2>
</example>

<example>
English:
<premise>The women met for coffee.</premise>
<choice1>The cafe reopened in a new location.</choice1>
<choice2>They wanted to catch up with each other.</choice2>
Hebrew:
<premise>הנשים נפגשו לקפה.</premise>
<choice1>בית הקפה נפתח מחדש במיקום חדש.</choice1>
<choice2>הן רצו להתעדכן אחת עם השנייה.</choice2>
</example>

<example>
English:
<premise>The guests of the party hid behind the couch.</premise>
<choice1>It was a surprise party.</choice1>
<choice2>It was a birthday party.</choice2>
Hebrew:
<premise>אורחי המסיבה התחבאו מאחורי הספה.</premise>
<choice1>זו הייתה מסיבת הפתעה.</choice1>
<choice2>זו הייתה מסיבת יום הולדת.</choice2>
</example>

<example>
English:
<premise>The stain came out of the shirt.</premise>
<choice1>I bleached the shirt.</choice1>
<choice2>I patched the shirt.</choice2>
Hebrew:
<premise>הכתם ירד מהחולצה.</premise>
<choice1>השתמשתי במלבין כביסה על החולצה.</choice1>
<choice2>תיקנתי את החולצה עם טלאי.</choice2>
</example>
</fewshot_examples>"""


COPA_FEW_SHOTS_EFFECT = """<fewshot_examples>
<example>
English:
<premise>The physician misdiagnosed the patient.</premise>
<choice1>The patient disclosed confidential information to the physician.</choice1>
<choice2>The patient filed a malpractice lawsuit against the physician.</choice2>
Hebrew:
<premise>הרופא אבחן את המטופל באבחנה שגויה.</premise>
<choice1>המטופל חשף מידע סודי לרופא.</choice1>
<choice2>המטופל הגיש תביעת רשלנות רפואית נגד הרופא.</choice2>
</example>

<example>
English:
<premise>The elderly woman suffered a stroke.</premise>
<choice1>The woman's daughter moved in to take care of her.</choice1>
<choice2>The woman's daughter came over to clean her house.</choice2>
Hebrew:
<premise>האישה הקשישה עברה שבץ מוחי.</premise>
<choice1>בתה של האישה עברה לגור איתה כדי לטפל בה.</choice1>
<choice2>בתה של האישה באה לנקות את ביתה.</choice2>
</example>

<example>
English:
<premise>The pond froze over for the winter.</premise>
<choice1>People brought boats to the pond.</choice1>
<choice2>People skated on the pond.</choice2>
Hebrew:
<premise>האגם קפא לקראת החורף.</premise>
<choice1>אנשים הביאו סירות לאגם.</choice1>
<choice2>אנשים החליקו על הקרח באגם.</choice2>
</example>

<example>
English:
<premise>The offender violated parole.</premise>
<choice1>She stole money from a church.</choice1>
<choice2>She was sent back to jail.</choice2>
Hebrew:
<premise>העבריין הפר את תנאי השחרור שלו.</premise>
<choice1>הוא גנב כסף.</choice1>
<choice2>הוא נשלח בחזרה לכלא.</choice2>
</example>

<example>
English:
<premise>I poured water on my sleeping friend.</premise>
<choice1>My friend awoke.</choice1>
<choice2>My friend snored.</choice2>
Hebrew:
<premise>שפכתי מים על חברי הישן.</premise>
<choice1>חברי התעורר.</choice1>
<choice2>חברי נחר.</choice2>
</example>
</fewshot_examples>"""
