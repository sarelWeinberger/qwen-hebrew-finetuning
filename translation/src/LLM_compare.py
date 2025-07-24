from src.call_models import call_gemini, all_string_gemini_config, all_list_gemini_config
from tqdm.notebook import tqdm
from time import time, sleep


CHOOSE_BEST_INSTRUCT_V1 = """Your task is to choose the best translation from English to Hebrew, given the original text and two different translations. Follow these guidelines:
1. The input is 'English: <key 1>text 1</key 1>\n<key 2>text 2</key 2>...\n\noption A: <key 1>translated text 1</key 1>\n<key 2>translated text 2</key 2>...\n\noption B: <key 1>translated text 1</key 1>\n<key 2>translated text 2</key 2>...'.
2. Choose the option which have the best fluency, while maintaining the same style and semantic meaning as the original.
3. The output should be either 'option A' or 'option B', according to the better translation."""


CHOOSE_BEST_INSTRUCT_V2 = """Your task is choose between two possible translatoins the better translation from English to Hebrew.
Choose the option which have the best fluency, while maintaining the same semantic meaning as the original.

<input format>
English: Original text\n\noption A: First Translation option\n\noption B: Second Translation option.
</input format>
"""

CHOOSE_BEST_INSTRUCT = """Your task is choose the best translation between two possible translatoins from English to Hebrew.
Choose the option which have the best fluency, while maintaining the same semantic meaning as the original.

<input format>
English:
{Original text}

First option:
{First Translation option}

Second option:
{Second Translation option}
</input format>"""


def compare_using_gemini(google_client, original, translation_1, translation_2):
    choose_model_config = all_string_gemini_config(
        ['Best Model'],
        CHOOSE_BEST_INSTRUCT,
        ['First option', 'Second option'],
        10_000,
        temp=1
    )
    compare_query = f'English: {original}\n\nFirst option: {translation_1}\n\nSecond option: {translation_2}'

    output = call_gemini(google_client, compare_query, choose_model_config)
    result = output.parsed
    if result['Best Model'] == 'First option':
        return (result['Best Model'], translation_1)
    else:
        return (result['Best Model'], translation_2)