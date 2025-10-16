"""
This file contains variables with prompts to be used when translating the 'HellaSwag' benchmark:
https://huggingface.co/datasets/hellaswag
"""


HELLASWAG_INSTRUCT_V1_GEMINI = """Your task is to translate the given English context, activity label, and possible endings into Hebrew. Follow these guidelines:

1. Translate only the context, activity label, and ending options provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the context and endings as accurately as possible in the Hebrew translation.
3. Maintain the same formatting as the original English version.
4. Write the translations in a clear and natural Hebrew style suitable for commonsense reasoning tasks.
5. Pay special attention to maintaining the logical flow and coherence of the context when translating.
6. Cultural adaptation for rare activities: If the activity or context involves something very rare or unfamiliar in Israeli culture (like lacrosse, specific American transit systems, or highly location-specific activities), perform transcreation instead of direct translation - replace with a similar Israeli equivalent that tests the same commonsense reasoning ability while maintaining the same logical structure and difficulty level.
7. Keep the "plausibility" of wrong answers: Ensure that incorrect answer choices remain tempting but less plausible than the correct answer in the Israeli context."""


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
<ending 1>מתחיל להסיר את הקרח מהשמשה הקדמית באמצעות שפכטל.</ending 1>
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
<ending 4>לשטוף את הרכב שלה בשביל החניה.</ending 4>
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


# ------------------------------------------------------------------------------------------------
# ------------------------------- HellaSwag Classification Prompts -------------------------------
# ------------------------------------------------------------------------------------------------

HELLASWAG_CLASSIFICATION_GEMINI = """You are an assistant helping translate benchmarks from English to Hebrew, for evaluation of LLMs.
Your task is to classify samples from the HellaSwag Dataset into three categories, representing how much the subject in the question is familiar to Israeli culture:
1. Universal - A universal subject is a subject that everybody knows and anyone can easily and fully understand, both in Israeli culture and other cultures. Those questions can be translated without changes at all.
2. Can be localized - A subject which is unfamiliar in Israeli culture, but not completely foreign to it. Those questions need a little explanation or adjustment during translation to be understood in Israeli culture.
3. Foreign - A Foreign subject, meaning the subject is foreign to Israeli culture. For example: cars aren't covered in ice in the Israeli winter, or Israelis don't play baseball."""

# 4. Verb outside of answers - When the vers in the question is in the 'ctx_b' part and not in the 'ending' options.
# 4. If the verb in the sample is in the 'ctx_b' part of the sample, and not in the 'ending i' options, classify the sample as a 'Verb outside of answer'. This makes the translation to hebrew harder, because sometimes the same verb can be translated differently according to the context. NOTICE that verbs in the 'ctx_a' part are completely fine.


HELLASWAG_CLASSIFICATION_FEW_SHOTS = """<fewshots_examples>
<example>
Question:
<activity_label>Getting a haircut</activity_label>
<ctx_a>The man in the blue shirt sits on the chair next to the sink. The other man begins washing his hair. He scrubs in the shampoo and then washes it off.</ctx_a>
<ctx_b>he</ctx_b>
<ctx>The man in the blue shirt sits on the chair next to the sink. The other man begins washing his hair. He scrubs in the shampoo and then washes it off. he</ctx>
<ending 1>then combs it and blow dries his hair after styling it with gel.</ending 1>
<ending 2>shows the razor that he has for shaving his hair.</ending 2>
<ending 3>hair is now dry, he is on his way to the barber.</ending 3>
<ending 4>moves the bucket to the other side of the sink and continues washing his hair.</ending 4>

Classification:
Universal
</example>

<example>
Question:
<activity_label>Playing drums</activity_label>
<ctx_a>A man is seen playing the drums with another man beside him with several lights flash around his face.</ctx_a>
<ctx_b>another man</ctx_b>
<ctx>A man is seen playing the drums with another man beside him with several lights flash around his face. another man</ctx>
<ending 1>is seen holding a guitar and several shots of a stage are shown.</ending 1>
<ending 2>plays a drum set while the camera follows around.</ending 2>
<ending 3>is seen speaking to the man playing drums and the man fades in and out.</ending 3>
<ending 4>is seen walking in and out of frame as the man plays with the drummer again and the man continues playing.</ending 4>

Classification:
Universal
</example>

<example>
Question:
<activity_label>Tai chi</activity_label>
<ctx_a>Three women are standing in a park in a field doing yoga.</ctx_a>
<ctx_b>man</ctx_b>
<ctx>Three women are standing in a park in a field doing yoga. man</ctx>
<ending 1>is running in the park behind the women.</ending 1>
<ending 2>is standing inside a gym talking to the camera and showing the exercises.</ending 2>
<ending 3>is sitting in the rack talking to the three women.</ending 3>
<ending 4>is talking to camera and take her robe off to reveal her beautiful legs, and is shown doing monyou pose.</ending 4>

Classification:
Can be localized
</example>

<example>
Question:
<activity_label>Cheerleading</activity_label>
<ctx_a>Half of the group kneels on the floor and the other standing group exits to the sides.</ctx_a>
<ctx_b>the remaining group</ctx_b>
<ctx>Half of the group kneels on the floor and the other standing group exits to the sides. the remaining group</ctx>
<ending 1>leaves and the two groups on stage speak.</ending 1>
<ending 2>stands back up and does a second dance routine together.</ending 2>
<ending 3>of people briefly arrives to the mats and kneel on the second row of the gym.</ending 3>
<ending 4>of group walk together up the stairs and through the exit doors together.</ending 4>

Classification:
Can be localized
</example>

<example>
Question:
<activity_label>Removing ice from car</activity_label>
<ctx_a>A man is sitting in a car. He starts the car and begins to drive. The car is covered in ice, so he</ctx_a>
<ctx_b>gets out and</ctx_b>
<ctx>A man is sitting in a car. He starts the car and begins to drive. The car is covered in ice, so he gets out and</ctx>
<ending 1>starts to remove the ice from the windshield using a scraper.</ending 1>
<ending 2>begins to dance on the sidewalk.</ending 2>
<ending 3>starts to cook breakfast on the hood of the car.</ending 3>
<ending 4>puts on his swimming suit.</ending 4>

Classification:
Foreign
</example>

<example>
Question:
<activity_label>Hitting a pinata</activity_label>
<ctx_a>A woman hands a young boy a stick with a little girl grabbing for her attention.</ctx_a>
<ctx_b>the pinata</ctx_b>
<ctx>A woman hands a young boy a stick with a little girl grabbing for her attention. the pinata</ctx>
<ending 1>is lifted and the boy begins swinging at the object.</ending 1>
<ending 2>pops, and a child in white attempts to hit the pinata, but fails.</ending 2>
<ending 3>pops out of the playground.</ending 3>
<ending 4>is hit one after the other, and the girl lets go her hold.</ending 4>

Classification:
Foreign
</example>
</fewshots_examples>"""

# <example>
# Question:
# <activity_label>Making a sandwich</activity_label>
# <ctx_a>A woman is in the kitchen preparing lunch. She takes out bread and various ingredients. She</ctx_a>
# <ctx_b>begins to</ctx_b>
# <ctx>A woman is in the kitchen preparing lunch. She takes out bread and various ingredients. She begins to</ctx>
# <ending 1>spread peanut butter on the bread slices.</ending 1>
# <ending 2>paint the kitchen walls.</ending 2>
# <ending 3>practice playing the piano.</ending 3>
# <ending 4>wash her car in the driveway.</ending 4>

# Classification:
# Verb outside of answers
# </example>

# <example>
# Question:
# <activity_label>Washing dishes</activity_label>
# <ctx_a>A person is standing at the kitchen sink after dinner. There are dirty dishes piled up. They</ctx_a>
# <ctx_b>turn on the</ctx_b>
# <ctx>A person is standing at the kitchen sink after dinner. There are dirty dishes piled up. They turn on the</ctx>
# <ending 1>water and start washing the dishes with soap.</ending 1>
# <ending 2>television to watch the news.</ending 2>
# <ending 3>lawn mower to cut the grass.</ending 3>
# <ending 4>computer to play video games.</ending 4>

# Classification:
# Verb outside of answers
# </example>