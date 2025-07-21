"""
This file contains variables with prompts to be used when translating the 'HellaSwag' benchmark:
https://huggingface.co/datasets/hellaswag

This file contains only the chosen prompts to be used.
The file 'old_hellaswag_prompts.py' contains old version prompts that we have tested.
"""


# Instruction prompts
HELLASWAG_INSTRUCT_V6_CLAUDE = """<instruction>
Your task is to translate the given English context, activity label, and possible endings into Hebrew. Follow these guidelines:

1. Translate only the context, activity label, and ending options provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the context and endings as accurately as possible in the Hebrew translation.
3. Maintain the same formatting as the original English version.
4. Write the translations in a clear and natural Hebrew style suitable for commonsense reasoning tasks.
5. Pay special attention to maintaining the logical flow and coherence of the context when translating.
6. Cultural adaptation for rare activities: If the activity or context involves something very rare or unfamiliar in Israeli culture (like lacrosse, specific American transit systems, or highly location-specific activities), perform transcreation instead of direct translation - replace with a similar Israeli equivalent that tests the same commonsense reasoning ability while maintaining the same logical structure and difficulty level.
7. Keep the "plausibility" of wrong answers: Ensure that incorrect answer choices remain tempting but less plausible than the correct answer in the Israeli context.
</instruction>"""

HELLASWAG_INSTRUCT_V1_CLAUDE_REFINE = """<instruction>
Your task is to translate the given English context, activity label, and possible endings into Hebrew. First write a possible translation and then improve it. 
Follow these guidelines:

1. Translate only the context, activity label, and ending options provided. Do not add any additional text.
2. Preserve the original semantic meaning and intent of the context and endings as accurately as possible in the Hebrew translation.
3. Maintain the same formatting as the original English version.
4. Write the translations in a clear and natural Hebrew style suitable for commonsense reasoning tasks.
5. Pay special attention to maintaining the logical flow and coherence of the context when translating.
6. Cultural adaptation for rare activities: If the activity or context involves something very rare or unfamiliar in Israeli culture (like lacrosse, specific American transit systems, or highly location-specific activities), perform transcreation instead of direct translation - replace with a similar Israeli equivalent that tests the same commonsense reasoning ability while maintaining the same logical structure and difficulty level.
7. Keep the "plausibility" of wrong answers: Ensure that incorrect answer choices remain tempting but less plausible than the correct answer in the Israeli context.
</instruction>"""

HELLASWAG_INSTRUCT_V1_CLAUDE_MULTI = """<instruction>
Your task is to translate the given English context, activity label, and possible endings into Hebrew. First, write three different translations for each field, and then choose the best translation of each field. Follow these guidelines:

1. Preserve the original semantic meaning and intent of the context and endings as accurately as possible in the Hebrew translation.
2. Maintain the same formatting as the original English version.
3. Write the translations in a clear and natural Hebrew style suitable for commonsense reasoning tasks.
4. Pay special attention to maintaining the logical flow and coherence of the context when translating.
5. Cultural adaptation for rare activities: If the activity or context involves something very rare or unfamiliar in Israeli culture (like lacrosse, specific American transit systems, or highly location-specific activities), perform transcreation instead of direct translation - replace with a similar Israeli equivalent that tests the same commonsense reasoning ability while maintaining the same logical structure and difficulty level.
6. Keep the "plausibility" of wrong answers: Ensure that incorrect answer choices remain tempting but less plausible than the correct answer in the Israeli context.
</instruction>"""

HELLASWAG_INSTRUCT_V1_GEMINI = """Your task is to translate the given English context, activity label, and possible endings into Hebrew. Follow these guidelines:

1. Translate only the context, activity label, and ending options provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the context and endings as accurately as possible in the Hebrew translation.
3. Maintain the same formatting as the original English version.
4. Write the translations in a clear and natural Hebrew style suitable for commonsense reasoning tasks.
5. Pay special attention to maintaining the logical flow and coherence of the context when translating.
6. Cultural adaptation for rare activities: If the activity or context involves something very rare or unfamiliar in Israeli culture (like lacrosse, specific American transit systems, or highly location-specific activities), perform transcreation instead of direct translation - replace with a similar Israeli equivalent that tests the same commonsense reasoning ability while maintaining the same logical structure and difficulty level.
7. Keep the "plausibility" of wrong answers: Ensure that incorrect answer choices remain tempting but less plausible than the correct answer in the Israeli context."""

HELLASWAG_INSTRUCT_V1_GEMINI_MULTI = """Your task is to translate the given English context, activity label, and possible endings into Hebrew. First, write three different translations for each field, and then choose the best translation of each field. Follow these guidelines:

1. Translate only the context, activity label, and ending options provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the context and endings as accurately as possible in the Hebrew translation.
3. Maintain the same formatting as the original English version.
4. Write the translations in a clear and natural Hebrew style suitable for commonsense reasoning tasks.
5. Pay special attention to maintaining the logical flow and coherence of the context when translating.
6. Cultural adaptation for rare activities: If the activity or context involves something very rare or unfamiliar in Israeli culture (like lacrosse, specific American transit systems, or highly location-specific activities), perform transcreation instead of direct translation - replace with a similar Israeli equivalent that tests the same commonsense reasoning ability while maintaining the same logical structure and difficulty level.
7. Keep the "plausibility" of wrong answers: Ensure that incorrect answer choices remain tempting but less plausible than the correct answer in the Israeli context."""


# Format prompts (to be used in claude's prompt)
HELLASWAG_FORMAT = """<response_format>
<activity_label>Translated activity label</activity_label>
<ctx_a>Translated context part A</ctx_a>
<ctx_b>Translated context part B</ctx_b>
<ctx>Translated full context</ctx>
<ending 1>Translated ending option 1</ending 1>
<ending 2>Translated ending option 2</ending 2>
<ending 3>Translated ending option 3</ending 3>
<ending 4>Translated ending option 4</ending 4>
</response_format>"""


# The few-shots samples
HELLASWAG_FEW_SHOTS = """<fewshot_examples>
<example>
English:
<activity_label>Removing ice from car</activity_label>
<ctx_a>A man is sitting in a car. He starts the car and begins to drive. The car is covered in ice, so he</ctx_a>
<ctx_b>gets out and</ctx_b>
<ctx>A man is sitting in a car. He starts the car and begins to drive. The car is covered in ice, so he gets out and</ctx>
<ending 1>starts to remove the ice from the windshield using a scraper.</ending 1>
<ending 2>begins to dance on the sidewalk.</ending 2>
<ending 3>starts to cook breakfast on the hood of the car.</ending 3>
<ending 4>puts on his swimming suit.</ending 4>
Hebrew:
<activity_label>הסרת קרח מרכב</activity_label>
<ctx_a>גבר יושב ברכב. הוא מתניע את הרכב ומתחיל לנסוע. הרכב מכוסה בקרח, אז הוא</ctx_a>
<ctx_b>יוצא ו</ctx_b>
<ctx>גבר יושב ברכב. הוא מתניע את הרכב ומתחיל לנסוע. הרכב מכוסה בקרח, אז הוא יוצא ו</ctx>
<ending 1>מתחיל להסיר את הקרח מהשמשה הקדמית באמצעות מגרד.</ending 1>
<ending 2>מתחיל לרקוד על המדרכה.</ending 2>
<ending 3>מתחיל לבשל ארוחת בוקר על מכסה המנוע של הרכב.</ending 3>
<ending 4>לובש את בגד הים שלו.</ending 4>
</example>

<example>
English:
<activity_label>Making a sandwich</activity_label>
<ctx_a>A woman is in the kitchen preparing lunch. She takes out bread and various ingredients. She</ctx_a>
<ctx_b>begins to</ctx_b>
<ctx>A woman is in the kitchen preparing lunch. She takes out bread and various ingredients. She begins to</ctx>
<ending 1>spread peanut butter on the bread slices.</ending 1>
<ending 2>paint the kitchen walls.</ending 2>
<ending 3>practice playing the piano.</ending 3>
<ending 4>wash her car in the driveway.</ending 4>
Hebrew:
<activity_label>הכנת כריך</activity_label>
<ctx_a>אישה נמצאת במטבח ומכינה ארוחת צהריים. היא מוציאה לחם ומרכיבים שונים. היא</ctx_a>
<ctx_b>מתחילה</ctx_b>
<ctx>אישה נמצאת במטבח ומכינה ארוחת צהריים. היא מוציאה לחם ומרכיבים שונים. היא מתחילה</ctx>
<ending 1>למרוח חמאת בוטנים על פרוסות הלחם.</ending 1>
<ending 2>לצבוע את קירות המטבח.</ending 2>
<ending 3>לתרגל נגינה בפסנתר.</ending 3>
<ending 4>לשטוף את הרכב שלה בשביל הגישה לחניה.</ending 4>
</example>

<example>
English:
<activity_label>Playing soccer</activity_label>
<ctx_a>Children are playing soccer in a field. The ball is kicked towards the goal. The goalkeeper</ctx_a>
<ctx_b>jumps to</ctx_b>
<ctx>Children are playing soccer in a field. The ball is kicked towards the goal. The goalkeeper jumps to</ctx>
<ending 1>catch the ball with his hands.</ending 1>
<ending 2>climb a nearby tree.</ending 2>
<ending 3>start reading a book.</ending 3>
<ending 4>begin cooking dinner.</ending 4>
Hebrew:
<activity_label>לשחק כדורגל</activity_label>
<ctx_a>ילדים משחקים כדורגל במגרש. הכדור נבעט לכיוון השער. השוער</ctx_a>
<ctx_b>קופץ כדי</ctx_b>
<ctx>ילדים משחקים כדורגל במגרש. הכדור נבעט לכיוון השער. השוער קופץ כדי</ctx>
<ending 1>לתפוס את הכדור בידיו.</ending 1>
<ending 2>לטפס על עץ סמוך.</ending 2>
<ending 3>להתחיל לקרוא ספר.</ending 3>
<ending 4>להתחיל לבשל ארוחת ערב.</ending 4>
</example>

<example>
English:
<activity_label>Washing dishes</activity_label>
<ctx_a>A person is standing at the kitchen sink after dinner. There are dirty dishes piled up. They</ctx_a>
<ctx_b>turn on the</ctx_b>
<ctx>A person is standing at the kitchen sink after dinner. There are dirty dishes piled up. They turn on the</ctx>
<ending 1>water and start washing the dishes with soap.</ending 1>
<ending 2>television to watch the news.</ending 2>
<ending 3>lawn mower to cut the grass.</ending 3>
<ending 4>computer to play video games.</ending 4>
Hebrew:
<activity_label>שטיפת כלים</activity_label>
<ctx_a>אדם עומד ליד כיור המטבח אחרי ארוחת הערב. יש ערימה של כלים מלוכלכים. הוא</ctx_a>
<ctx_b>פותח את</ctx_b>
<ctx>אדם עומד ליד כיור המטבח אחרי ארוחת הערב. יש ערימה של כלים מלוכלכים. הוא פותח את</ctx>
<ending 1>הברז ומתחיל לשטוף את הכלים עם סבון.</ending 1>
<ending 2>הטלוויזיה כדי לצפות בחדשות.</ending 2>
<ending 3>מכסחת הדשא כדי לקצוץ את הדשא.</ending 3>
<ending 4>המחשב כדי לשחק משחקי וידאו.</ending 4>
</example>

<example>
English:
<activity_label>Brushing teeth</activity_label>
<ctx_a>A child is getting ready for bed. They go to the bathroom and pick up their toothbrush. They</ctx_a>
<ctx_b>squeeze the</ctx_b>
<ctx>A child is getting ready for bed. They go to the bathroom and pick up their toothbrush. They squeeze the</ctx>
<ending 1>toothpaste onto the brush and start brushing their teeth.</ending 1>
<ending 2>orange juice into a glass.</ending 2>
<ending 3>pillow to make it more comfortable.</ending 3>
<ending 4>bicycle tire to check the pressure.</ending 4>
Hebrew:
<activity_label>צחצוח שיניים</activity_label>
<ctx_a>ילד מתכונן לשינה. הוא הולך לשירותים ולוקח את מברשת השיניים שלו. הוא</ctx_a>
<ctx_b>סוחט את</ctx_b>
<ctx>ילד מתכונן לשינה. הוא הולך לשירותים ולוקח את מברשת השיניים שלו. הוא סוחט את</ctx>
<ending 1>משחת השיניים על המברשת ומתחיל לצחצח את השיניים.</ending 1>
<ending 2>מיץ התפוזים לתוך כוס.</ending 2>
<ending 3>הכרית כדי לעשות אותה נוחה יותר.</ending 3>
<ending 4>הצמיג של האופניים כדי לבדוק את הלחץ.</ending 4>
</example>
</fewshot_examples>"""
