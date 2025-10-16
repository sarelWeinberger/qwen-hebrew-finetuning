"""
This file contains variables with prompts that were used to translate the 'ARC_AI2' benchmark.
In qualitative testing, the final prompts peformed better than these prompts.

The file 'arc_prompts.py' contains the final prompts
"""

# Instruction prompts
"""
manual comparison on claude:
ARC_INSTRUCT_V1 < ARC_INSTRUCT_V2
ARC_INSTRUCT_V2 < ARC_INSTRUCT_V3
ARC_INSTRUCT_V4_CLAUDE < ARC_INSTRUCT_V3
ARC_INSTRUCT_V3 < ARC_INSTRUCT_V5_CLAUDE
ARC_INSTRUCT_V5_CLAUDE < ARC_INSTRUCT_V6_CLAUDE

manual comparison on Gemini 2.5:
Looks like ‘ARC_INSTRUCT_V1_GEMINI’ as a system prompt works best both
on gemini-flash and gemini-pro.

Enabling thinking improves results on the flash model, but only slightly
improves results on the pro model.
"""

ARC_INSTRUCT_V1 = """Your task is to translate the given English question and possible answers into Hebrew. Follow these guidelines:

1. Only translate the question and answer options provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the question and answers as accurately as possible in the Hebrew english_to_hebrew_translation.
3. Maintain the same formatting as the original English version, including any bullet points, numbering, or other formatting elements.
4. Provide the Hebrew english_to_hebrew_translation immediately after these instructions, without any preamble or additional context."""

ARC_INSTRUCT_V2 = """Your task is to translate the given English question and possible answers into Hebrew. Follow these guidelines:

1. Only translate the question and answer options provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the question and answers as accurately as possible in the Hebrew english_to_hebrew_translation.
3. Maintain the same formatting as the original English version, including any bullet points, numbering, or other formatting elements.
4. Write the translations in the style of grade school-level science questions.
5. Provide the Hebrew english_to_hebrew_translation immediately after these instructions, without any preamble or additional context."""

ARC_INSTRUCT_V3 = """Your task is to translate the given English question and possible answers into Hebrew. Follow these guidelines:

1. Only translate the question and answer options provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the question and answers as accurately as possible in the Hebrew english_to_hebrew_translation.
3. Maintain the same formatting as the original English version.
4. Write the translations in the style of grade school-level science questions.
5. Provide the Hebrew english_to_hebrew_translation immediately after these instructions, without any preamble or additional context."""

ARC_INSTRUCT_V4_CLAUDE = """Your task is to translate the given English question and possible answers into Hebrew. Follow these guidelines:

<instruction>
1. Only translate the question and answer options provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the question and answers as accurately as possible in the Hebrew english_to_hebrew_translation.
3. Maintain the same formatting as the original English version.
4. Write the translations in the style of grade school-level science questions.
</instruction>"""

ARC_INSTRUCT_V5_CLAUDE = """<instruction>
Your task is to translate the given English question and possible answers into Hebrew. Follow these guidelines:

1. Only translate the question and answer options provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the question and answers as accurately as possible in the Hebrew english_to_hebrew_translation.
3. Maintain the same formatting as the original English version.
4. Write the translations in the style of grade school-level science questions.
</instruction>"""

ARC_SYSTEM_V2_GEMINI = """You are an expert translator specializing in educational content. Your task is to translate grade-level science questions from English to Hebrew for the ARC_AI2 benchmark. Your primary audience is Israeli grade school students (grade 6)."""

ARC_INSTRUCT_V2_GEMINI = """Translate the following English science question and answers into Hebrew.

**Key Guidelines:**

1.  **Target Audience:** The english_to_hebrew_translation must be clear, natural, and scientifically accurate for an Israeli middle school student (תלמיד/ת חטיבת ביניים). Use terminology that is standard in the Israeli science curriculum for this age group.
2.  **Accuracy:** Preserve the precise scientific meaning of the original text. Do not simplify the core concept, but ensure the language used to explain it is accessible.
3.  **Grammar:** Pay close attention to Hebrew grammar, especially gender and number agreement (e.g., `דלת המקרר מוליכה` vs. `הברזל מוליך`).
4.  **Conciseness:** Be direct and to the point. Avoid overly academic or verbose phrasing.
5.  **Purity:** Translate *only* the text within the `<question>` and `<option>` tags. Do not add any introductory phrases, explanations, or text outside the required format."""

ARC_INSTRUCT_V3_GEMINI = """Translate the following English science question and answers into Hebrew.

Key Guidelines:

1. Target Audience: The english_to_hebrew_translation must be clear, natural, and scientifically accurate for an Israeli grade school student (תלמיד/ת יסודי). Use terminology that is standard in the Israeli science curriculum for this age group.
2. Accuracy: Preserve the precise scientific meaning of the original text. Do not simplify the core concept, but ensure the language used to explain it is accessible.
3. Grammar: Pay close attention to Hebrew grammar, especially gender and number agreement (e.g., `דלת המקרר מוליכה` vs. `הברזל מוליך`).
4. **Conciseness:** Be direct and to the point. Avoid overly academic or verbose phrasing.
5. Purity: Translate *only* the text within the `<question>` and `<option>` tags. Do not add any introductory phrases, explanations, or text outside the required format."""


# Format prompts (to be used in claude's prompt)
ARC_FORMAT_V4_V5 = """<response_format>
<question>Translated question</question>
<option 1>Translated answer option 1</option 1>
<option 2>Translated answer option 2</option 2>
<option 3>Translated answer option 3</option 3>
<option 4>Translated answer option 4</option 4>
</response_format>

Provide the Hebrew english_to_hebrew_translation immediately after these instructions, without any preamble or additional context."""


# Multiple english_to_hebrew_translation prompts
ARC_INSTRUCT_MULTI_V1 = """Your task is to translate the given English question and possible answers into possible Hebrew translations. Follow these guidelines:

1. Only translate the question and answer options provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the question and answers as accurately as possible in the Hebrew english_to_hebrew_translation.
3. Maintain the same formatting and style as the original English version.
4. Provide {X} possible translations for the question and each one of the answers.
5. Provide the Hebrew english_to_hebrew_translation immediately after these instructions, without any preamble or additional context."""