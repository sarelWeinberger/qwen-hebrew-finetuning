from src.call_models import call_claude_bedrock
from src.call_models import call_gemini, all_string_gemini_config, all_list_gemini_config
from datasets import Dataset
import re
from tqdm.notebook import tqdm
from time import time, sleep

# The base prompt of the English-Hebrew
BASE_PROMPT = "English:\n{X}\nHebrew:\n{Y}"

CHOOSE_INSTRUCT = """Your task is to choose the best translation from English to Hebrew, given a number of options. Follow these guidelines:
1. The input is <key>English | [option 1, option 2, ....]</key>, with number of 'keys'.
2. Choose the option which have the best fluency, while maintaining the same style and semantic meaning as the original.
3. The output for 'key' should be the chosen best translation from English of the 'key'"""

CHOOSE_MODEL_INSTRUCT = """Your task is to choose the best translation from English to Hebrew, given two options of two different models. Follow these guidelines:
1. The input is 'English: <key>text</key>\n\noption A: <key>translated text</key>\n\noption B: <key>translated text</key>'.
2. Choose the option which have the best fluency, while maintaining the same style and semantic meaning as the original.
3. The output should be either 'option A' or 'option B', according to the better translation."""


def dict_to_prompt(dct):
    return '\n'.join([f'<{k}>{dct[k]}</{k}>' for k in dct])


def claude_translation(bedrock_client, datasets, instruct, few_shots, sample_format, sample_to_dict, dict_to_sample):
    """
    Translate all given datasets using one of the claude famaliy models.
    """
    # The final prompt for the models
    final_prompt = few_shots + '\n\n' + sample_format + '\n\n'
    # final_prompt = instruct + '\n\n' + few_shots + '\n\n' + sample_format + '\n\n'
    hebrew_dataset = {}
    text_output = {}
    # Run on the different splits in the dataset
    for key in datasets:
        print(f'Translating {key}...')
        hebrew_dataset[key] = []
        text_output[key] = []
        # Run on all the split's samples
        for sample in tqdm(datasets[key], total=datasets[key].num_rows):
            # from sample to dict
            dct = sample_to_dict(sample)

            # Enter into prompt
            samples_query = dict_to_prompt(dct)
            query = final_prompt + BASE_PROMPT.format(X=samples_query, Y='')

            # Call claude
            output = call_claude_bedrock(bedrock_client, query, system_prompt=instruct)

            # Parse the model's output
            pattern = r"<(?!response_format\b)([^>]+)>(.*?)</\1>"
            matches = re.findall(pattern, output, re.DOTALL)
            # In case 'matches' contain two (or more) pairs with the same 'key', the
            # last value is the one that will be stored
            # In matches the order of the pairs is according to the appearance in 'output'
            result = {key: value.strip() for key, value in matches}
            if len(set(dct.keys()) - set(result.keys())) > 0:
                print(dct.keys())
                print(result.keys())
                print()
                print(output)
                print('NONONONONONO')
                return hebrew_dataset, text_output

            # Create New sample
            new_sample = dict_to_sample(sample, result)
            hebrew_dataset[key].append(new_sample)
            text_output[key].append(output)
            # print(output)
            # print()

        hebrew_dataset[key] = Dataset.from_list(hebrew_dataset[key])
    return hebrew_dataset, text_output
    
    
def get_gemini_thoughts_summary(response):
    thinking_summary = []
    for part in response.candidates[0].content.parts:
        if not part.text:
            continue
        if part.thought:
            thinking_summary.append(part.text)
    thinking_summary = '\n\n'.join(thinking_summary)
    return thinking_summary


def gemini_translation(google_client, datasets, instruct, few_shots, sample_to_dict, dict_to_sample, if_pro=False, think_bud=-1):
    """
    Translate all given datasets using one of the Gemini famaliy models.
    """
    # The final prompt for the models
    final_prompt = few_shots + '\n\n'
    hebrew_dataset = {}
    text_output = {}
    if if_pro:
        quota_min = 5
    else:
        quota_min = 10
        
    # Run on the different splits in the dataset
    start_time = time()
    for key in datasets:
        fields = sample_to_dict(datasets[key][0]).keys()
        config = all_string_gemini_config(fields, instruct, think_bud=think_bud)

        print(f'Translating {key}...')
        hebrew_dataset[key] = []
        text_output[key] = []
        # Run on all the split's samples
        for cnt, sample in enumerate(tqdm(datasets[key], total=datasets[key].num_rows), start=1):
            # from sample to dict
            dct = sample_to_dict(sample)

            # Enter into prompt
            samples_query = dict_to_prompt(dct)
            query = final_prompt + BASE_PROMPT.format(X=samples_query, Y='')

            # Call gemini
            do_it = True
            while do_it:
                try:
                    output = call_gemini(google_client, query, config, if_pro=if_pro)
                    do_it = False
                except:
                    sleep(10)

            # Parse the model's output
            thinking_summary = get_gemini_thoughts_summary(output)
            result = output.parsed

            # Create New sample
            new_sample = dict_to_sample(sample, result)
            hebrew_dataset[key].append(new_sample)
            text_output[key].append(thinking_summary)

            if cnt % quota_min == 0:
                passed = time() - start_time
                if passed < 60:
                    print('\r' + ' ' * 50 + '\rSleeping.... ', end='')
                    sleep(61 - passed)
                    print('Done!', end='')
                start_time = time()

        hebrew_dataset[key] = Dataset.from_list(hebrew_dataset[key])
    return hebrew_dataset, text_output


def options_to_prompt(original, hebrew):
    # return '\n'.join([f'<{k}>{original[k]} | [' + ', '.join(hebrew[k]) + f']</{k}>' for k in original])
    return '\n'.join([f'<{k}>{original[k]} | {hebrew[k]}</{k}>' for k in original])


# def gemini_multi_translation(google_client, datasets, instruct, few_shots, sample_to_dict, dict_to_sample, length=3, think_bud=-1):
#     """
#     Translate all given datasets using one of the Gemini famaliy models.
#     Ask Gemini to give a number of different options, and choose the best one.
#     """
#     assert length > 1, 'Numer of options must be greater then 1.'
#     # The final prompt for the models
#     final_prompt = few_shots + '\n\n'
#     hebrew_dataset = {}
#     # Run on the different splits in the dataset
#     for key in datasets:
#         fields = sample_to_dict(datasets[key][0]).keys()
#         config = all_list_gemini_config(fields, instruct.format(X=length), length, think_bud=think_bud)
#         choose_config = all_string_gemini_config(fields, CHOOSE_INSTRUCT)

#         print(f'Translating {key}...')
#         hebrew_dataset[key] = []
#         # Run on all the split's samples
#         for sample in tqdm(datasets[key], total=datasets[key].num_rows):
#             # from sample to dict
#             dct = sample_to_dict(sample)

#             # Enter into prompt
#             samples_query = dict_to_prompt(dct)
#             query = final_prompt + BASE_PROMPT.format(X=samples_query, Y='')

#             # Call gemini
#             output = call_gemini(google_client, query, config)

#             # Parse the model's output
#             result = output.parsed
#             # return dct, result 

#             # Choose best option
#             multi_query = options_to_prompt(dct, result)
#             output = call_gemini(google_client, multi_query, choose_config)
#             result = output.parsed

#             # Create New sample
#             new_sample = dict_to_sample(sample, result)
#             hebrew_dataset[key].append(new_sample)

#         hebrew_dataset[key] = Dataset.from_list(hebrew_dataset[key])
#     return hebrew_dataset


def gemini_claude_best_translation(
    google_client,
    bedrock_client,
    datasets,
    instruct,
    few_shots,
    sample_format,
    sample_to_dict,
    dict_to_sample
):
    """
    Translate using Gemini and Claude, then use Gemini to choose the best option for each translation.
    The best options is for the WHOLE sample, not separated for each field in the sample.
    """
    claude_final_prompt = instruct + '\n\n' + few_shots + '\n\n' + sample_format + '\n\n'
    gemini_final_prompt = few_shots + '\n\n'
    hebrew_dataset = {}
    # Run on the different splits in the dataset
    for key in datasets:
        fields = sample_to_dict(datasets[key][0]).keys()
        config = all_string_gemini_config(fields, instruct)
        choose_model_config = all_string_gemini_config(
            ['Best Model'],
            CHOOSE_MODEL_INSTRUCT,
            ['option A', 'option B']
        )
        
        print(f'Translating {key}...')
        hebrew_dataset[key] = []
        # Run on all the split's samples
        for sample in tqdm(datasets[key], total=datasets[key].num_rows):
            # from sample to dict
            dct = sample_to_dict(sample)

            # Enter into prompt
            samples_query = dict_to_prompt(dct)
            claude_query = claude_final_prompt + BASE_PROMPT.format(X=samples_query, Y='')
            gemini_query = gemini_final_prompt + BASE_PROMPT.format(X=samples_query, Y='')

            # Call claude
            claude_output = call_claude_bedrock(bedrock_client, claude_query)
            pattern = r"<([^>]+)>(.*?)</\1>"
            matches = re.findall(pattern, claude_output, re.DOTALL)
            claude_result = {key: value.strip() for key, value in matches}

            # Call gemini
            gemini_output = call_gemini(google_client, gemini_query, config)
            gemini_result = gemini_output.parsed
            
            # Choose best betweeb the two
            gemini_tran_query = dict_to_prompt(gemini_result)
            claude_tran_query = dict_to_prompt(claude_result)
            compare_query = f'English: {samples_query}\n\noption A: {gemini_tran_query}\n\noption B: {claude_tran_query}'
            
            output = call_gemini(google_client, compare_query, choose_model_config)
            result = output.parsed
            if result['Best Model'] == 'option A':
                new_sample = dict_to_sample(sample, gemini_result)
                new_sample['Translated Model'] = 'Gemini'
            else:
                new_sample = dict_to_sample(sample, claude_result)
                new_sample['Translated Model'] = 'Claude'

            # Create New sample
            hebrew_dataset[key].append(new_sample)

        hebrew_dataset[key] = Dataset.from_list(hebrew_dataset[key])
    return hebrew_dataset