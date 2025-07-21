"""
This file contains variables with prompts to be used when translating the 'MMLU' benchmark:
https://huggingface.co/datasets/cais/mmlu

This file contains only the chosen prompts to be used.
The file 'old_mmlu_prompts.py' contains old version prompts that we have tested.
"""


# Instruction prompts
MMLU_INSTRUCT_V6_CLAUDE = """<instruction>
Your task is to translate the given English multiple-choice question, subject, and answer options into Hebrew. Follow these guidelines:

1. Translate only the question, subject, and answer choices provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the question and answer choices as accurately as possible in the Hebrew translation.
3. Maintain the same formatting as the original English version.
4. Write the translations in a clear and natural Hebrew style suitable for academic and knowledge assessment tasks.
5. Pay special attention to maintaining the logical flow and coherence of the question when translating.
6. Ensure that specialized terminology from various academic fields is translated accurately while maintaining Hebrew academic conventions.
7. Distinguish between universal and location-specific content: For universal subjects (mathematics, biology, physics, chemistry, etc.), perform direct translation with accurate terminology. For location-specific subjects (American history, American law, specific regional contexts), create parallel questions based on Israeli context that test the same knowledge and skill level.
8. Maintain academic complexity: Ensure that Hebrew phrasing preserves the academic complexity and precision of the original question.
9. Professional terminology accuracy: Use precise Hebrew academic and professional terminology that matches Israeli academic standards in each field.
</instruction>"""

MMLU_INSTRUCT_V1_CLAUDE_REFINE = """<instruction>
Your task is to translate the given English multiple-choice question, subject, and answer options into Hebrew. First write a possible translation and then improve it.
Follow these guidelines:

1. Translate only the question, subject, and answer choices provided. Do not add any additional text.
2. Preserve the original semantic meaning and intent of the question and answer choices as accurately as possible in the Hebrew translation.
3. Maintain the same formatting as the original English version.
4. Write the translations in a clear and natural Hebrew style suitable for academic and knowledge assessment tasks.
5. Pay special attention to maintaining the logical flow and coherence of the question when translating.
6. Ensure that specialized terminology from various academic fields is translated accurately while maintaining Hebrew academic conventions.
7. Distinguish between universal and location-specific content: For universal subjects (mathematics, biology, physics, chemistry, etc.), perform direct translation with accurate terminology. For location-specific subjects (American history, American law, specific regional contexts), create parallel questions based on Israeli context that test the same knowledge and skill level.
8. Maintain academic complexity: Ensure that Hebrew phrasing preserves the academic complexity and precision of the original question.
9. Professional terminology accuracy: Use precise Hebrew academic and professional terminology that matches Israeli academic standards in each field.
</instruction>"""

MMLU_INSTRUCT_V1_CLAUDE_MULTI = """<instruction>
Your task is to translate the given English multiple-choice question, subject, and answer options into Hebrew. First, write three different translations for each field, and then choose the best translation of each field. Follow these guidelines:

1. Preserve the original semantic meaning and intent of the question and answer choices as accurately as possible in the Hebrew translation.
2. Maintain the same formatting as the original English version.
3. Write the translations in a clear and natural Hebrew style suitable for academic and knowledge assessment tasks.
4. Pay special attention to maintaining the logical flow and coherence of the question when translating.
5. Ensure that specialized terminology from various academic fields is translated accurately while maintaining Hebrew academic conventions.
6. Distinguish between universal and location-specific content: For universal subjects (mathematics, biology, physics, chemistry, etc.), perform direct translation with accurate terminology. For location-specific subjects (American history, American law, specific regional contexts), create parallel questions based on Israeli context that test the same knowledge and skill level.
7. Maintain academic complexity: Ensure that Hebrew phrasing preserves the academic complexity and precision of the original question.
8. Professional terminology accuracy: Use precise Hebrew academic and professional terminology that matches Israeli academic standards in each field.
</instruction>"""

MMLU_INSTRUCT_V1_GEMINI = """Your task is to translate the given English multiple-choice question, subject, and answer options into Hebrew. Follow these guidelines:

1. Translate only the question, subject, and answer choices provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the question and answer choices as accurately as possible in the Hebrew translation.
3. Maintain the same formatting as the original English version.
4. Write the translations in a clear and natural Hebrew style suitable for academic and knowledge assessment tasks.
5. Pay special attention to maintaining the logical flow and coherence of the question when translating.
6. Ensure that specialized terminology from various academic fields is translated accurately while maintaining Hebrew academic conventions.
7. Distinguish between universal and location-specific content: For universal subjects (mathematics, biology, physics, chemistry, etc.), perform direct translation with accurate terminology. For location-specific subjects (American history, American law, specific regional contexts), create parallel questions based on Israeli context that test the same knowledge and skill level.
8. Maintain academic complexity: Ensure that Hebrew phrasing preserves the academic complexity and precision of the original question.
9. Professional terminology accuracy: Use precise Hebrew academic and professional terminology that matches Israeli academic standards in each field."""

MMLU_INSTRUCT_V1_GEMINI_MULTI = """Your task is to translate the given English multiple-choice question, subject, and answer options into Hebrew. First, write three different translations for each field, and then choose the best translation of each field. Follow these guidelines:

1. Translate only the question, subject, and answer choices provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the question and answer choices as accurately as possible in the Hebrew translation.
3. Maintain the same formatting as the original English version.
4. Write the translations in a clear and natural Hebrew style suitable for academic and knowledge assessment tasks.
5. Pay special attention to maintaining the logical flow and coherence of the question when translating.
6. Ensure that specialized terminology from various academic fields is translated accurately while maintaining Hebrew academic conventions.
7. Distinguish between universal and location-specific content: For universal subjects (mathematics, biology, physics, chemistry, etc.), perform direct translation with accurate terminology. For location-specific subjects (American history, American law, specific regional contexts), create parallel questions based on Israeli context that test the same knowledge and skill level.
8. Maintain academic complexity: Ensure that Hebrew phrasing preserves the academic complexity and precision of the original question.
9. Professional terminology accuracy: Use precise Hebrew academic and professional terminology that matches Israeli academic standards in each field."""


# Format prompts (to be used in claude's prompt)
MMLU_FORMAT = """<response_format>
<question>Translated question</question>
<choice_a>Translated option A</choice_a>
<choice_b>Translated option B</choice_b>
<choice_c>Translated option C</choice_c>
<choice_d>Translated option D</choice_d>
</response_format>"""


# The few-shots samples
MMLU_FEW_SHOTS = """<fewshot_examples>
<example>
English:
<subject>anatomy</subject>
<question>What is the embryological origin of the hyoid bone?</question>
<choice_a>The first pharyngeal arch</choice_a>
<choice_b>The first and second pharyngeal arches</choice_b>
<choice_c>The second pharyngeal arch</choice_c>
<choice_d>The second and third pharyngeal arches</choice_d>
Hebrew:
<subject>אנטומיה</subject>
<question>מהו המקור האמבריולוגי של עצם הלשון?</question>
<choice_a>קשת הלוע הראשונה</choice_a>
<choice_b>קשתות הלוע הראשונה והשנייה</choice_b>
<choice_c>קשת הלוע השנייה</choice_c>
<choice_d>קשתות הלוע השנייה והשלישית</choice_d>
</example>

<example>
English:
<subject>abstract_algebra</subject>
<question>Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.</question>
<choice_a>0</choice_a>
<choice_b>4</choice_b>
<choice_c>2</choice_c>
<choice_d>6</choice_d>
Hebrew:
<subject>אלגברה מופשטת</subject>
<question>מצאו את המעלה עבור הרחבת השדה הנתונה Q(sqrt(2), sqrt(3), sqrt(18)) מעל Q.</question>
<choice_a>0</choice_a>
<choice_b>4</choice_b>
<choice_c>2</choice_c>
<choice_d>6</choice_d>
</example>

<example>
English:
<subject>high_school_biology</subject>
<question>Which of the following is responsible for the cohesion of water molecules?</question>
<choice_a>Ionic bonds</choice_a>
<choice_b>Hydrogen bonds</choice_b>
<choice_c>Covalent bonds</choice_c>
<choice_d>Van der Waals forces</choice_d>
Hebrew:
<subject>ביולוגיה ברמת תיכון</subject>
<question>איזה מהבאים אחראי ללכידות מולקולות המים?</question>
<choice_a>קשרים יוניים</choice_a>
<choice_b>קשרי מימן</choice_b>
<choice_c>קשרים קוולנטיים</choice_c>
<choice_d>כוחות ואן דר ואלס</choice_d>
</example>

<example>
English:
<subject>philosophy</subject>
<question>According to Kant, the moral worth of an action depends on</question>
<choice_a>its consequences</choice_a>
<choice_b>the motivation behind it</choice_b>
<choice_c>social conventions</choice_c>
<choice_d>personal happiness</choice_d>
Hebrew:
<subject>פילוסופיה</subject>
<question>:לפי קאנט, הערך המוסרי של פעולה תלוי ב</question>
<choice_a>ההשלכות שלה</choice_a>
<choice_b>המניע שמאחוריה</choice_b>
<choice_c>מוסכמות חברתיות</choice_c>
<choice_d>אושר אישי</choice_d>
</example>

<example>
English:
<subject>computer_security</subject>
<question>Which of the following is a type of malware that replicates itself?</question>
<choice_a>Trojan horse</choice_a>
<choice_b>Virus</choice_b>
<choice_c>Spyware</choice_c>
<choice_d>Adware</choice_d>
Hebrew:
<subject>אבטחת מחשבים</subject>
<question>איזה מהבאים הוא סוג של תוכנה זדונית שמשכפלת את עצמה?</question>
<choice_a>סוס טרויאני</choice_a>
<choice_b>וירוס</choice_b>
<choice_c>תוכנת ריגול</choice_c>
<choice_d>תוכנת פרסומות</choice_d>
</example>
</fewshot_examples>"""
