from src.call_models import call_claude_bedrock, call_claude_bedrock_with_thinking
from src.call_models import call_gemini, all_string_gemini_config, all_list_gemini_config
from datasets import Dataset
import re
from tqdm.auto import tqdm
from time import time, sleep

import botocore.exceptions

import pickle

# The base prompt of the English-Hebrew
BASE_PROMPT = "English:\n{X}\nHebrew:\n{Y}"


def dict_to_prompt(dct):
    # Don't add the 'translation_status' value to the string...
    return '\n'.join([f'<{k}>{dct[k]}</{k}>' for k in dct if k != 'translation_status'])


def claude_translation(bedrock_client, datasets, instruct, few_shots, sample_format, sample_to_dict, dict_to_sample, if_four=False):
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
            # print(query)
            # print('\n\n\n\n')
            # print(instruct)
            # exit()
            try:
                # output = call_claude_bedrock(bedrock_client, query, system_prompt=instruct)
                output, thinking = call_claude_bedrock_with_thinking(bedrock_client, query, system_prompt=instruct, if_four=if_four)
            except botocore.exceptions.ReadTimeoutError as e:
                print("Read timeout occurred:", e)
                print('Droped sample number: ', len(hebrew_dataset[key]))

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
            # text_output[key].append(output)
            text_output[key].append('Thinking:\n' + thinking + '\n\nText:\n' + output)
            # print(output)
            # print()

            cur_size = len(hebrew_dataset[key])
            if cur_size % 25 == 0:
                cp_name = f'checkpoints/claude_{key}_{cur_size}.pkl'
                cp_name_text = f'checkpoints/claude_{key}_{cur_size}_text.pkl'
                with open(cp_name, 'wb') as f:
                    pickle.dump(hebrew_dataset[key], f)
                with open(cp_name_text, 'wb') as f:
                    pickle.dump(text_output[key], f)

        hebrew_dataset[key] = Dataset.from_list(hebrew_dataset[key])
    return hebrew_dataset, text_output


def get_gemini_thoughts_summary(response):
    thinking_summary = []

    # Safeguard: check if 'candidates' exists and is valid
    if not hasattr(response, 'candidates') or not response.candidates:
        return ""

    candidate = response.candidates[0]

    if not hasattr(candidate, 'content') or not candidate.content:
        return ""

    if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
        return ""

    for part in candidate.content.parts:
        # Defensive check: ensure part has 'text'
        if hasattr(part, 'text') and part.text:
            if hasattr(part, 'thought') and part.thought:
                thinking_summary.append(part.text)

    return '\n\n'.join(thinking_summary)


def gemini_translation(google_client, datasets, instruct, few_shots, sample_to_dict, dict_to_sample, if_pro=False,
                       think_bud=-1):
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
                    print('-', end='')
                    output = call_gemini(google_client, query, config, if_pro=if_pro)
                    print('|', end='')
                    do_it = False
                except Exception as e:
                    print('\r' + ' ' * 50 + f'\rSleeping in the "While" because of {e}.... ', end='')
                    sleep(10)
                    print('Done!', end='')

            # Parse the model's output
            thinking_summary = get_gemini_thoughts_summary(output)
            result = output.parsed

            # Create New sample - ADD ERROR HANDLING HERE
            try:
                if result is None:
                    # Handle case where parsing failed
                    print(f"Warning: Failed to parse response for sample {cnt - 1}, skipping...")
                    print(output)
                    # return None

                    # Create a fallback sample with original data and failure indicator
                    new_sample = sample.copy()
                    # Add a field to indicate english_to_hebrew_translation failure
                    if 'translation_status' not in new_sample:
                        new_sample['translation_status'] = 'Failed to translate!'

                    hebrew_dataset[key].append(new_sample)
                    text_output[key].append(thinking_summary)
                    continue

                new_sample = dict_to_sample(sample, result)
                # Mark as successfully translated
                if 'translation_status' not in new_sample:
                    new_sample['translation_status'] = 'Success'

                hebrew_dataset[key].append(new_sample)
                text_output[key].append(thinking_summary)

            except Exception as e:
                # Handle any other errors in dict_to_sample
                print(f"Error processing sample {cnt - 1}: {e}")

                # Create a fallback sample
                new_sample = sample.copy()
                new_sample['translation_status'] = 'Failed to translate!'

                hebrew_dataset[key].append(new_sample)
                text_output[key].append(thinking_summary)

            cur_size = len(hebrew_dataset[key])
            if cur_size % 15 == 0:
                cp_name = f'gemini_cp/gemini_{key}_{cur_size}.pkl'
                cp_name_text = f'gemini_cp/gemini_{key}_{cur_size}_text.pkl'
                with open(cp_name, 'wb') as f:
                    pickle.dump(hebrew_dataset[key], f)
                with open(cp_name_text, 'wb') as f:
                    pickle.dump(text_output[key], f)

        hebrew_dataset[key] = Dataset.from_list(hebrew_dataset[key])
    return hebrew_dataset, text_output


HELLASWAG_CLASSES = [
    'Universal',
    'Can be localized',
    'Foreign',
    # 'Verb outside',
]
CLASSIFICATION_PROMPT = "Question:\n{X}\nClassification:\n{Y}"


def gemini_classification(google_client, datasets, instruct, few_shots, sample_to_dict, dict_to_sample, think_bud=-1):
    """
    Translate all given datasets using one of the Gemini famaliy models.
    """
    # The final prompt for the models
    final_prompt = few_shots + '\n\n'
    final_labels = {}
    text_output = {}

    for key in datasets:
        config = all_string_gemini_config(['classification'], instruct, HELLASWAG_CLASSES, think_bud=think_bud)

        print(f'Classifying {key}...')
        final_labels[key] = []
        text_output[key] = []
        # Run on all the split's samples
        for cnt, sample in enumerate(tqdm(datasets[key], total=datasets[key].num_rows)):
            # from sample to dict
            dct = sample_to_dict(sample)

            # Enter into prompt
            samples_query = dict_to_prompt(dct)
            query = final_prompt + CLASSIFICATION_PROMPT.format(X=samples_query, Y='')

            # Call gemini
            do_it = True
            while do_it:
                try:
                    output = call_gemini(google_client, query, config, if_pro=False)
                    do_it = False
                except Exception as e:
                    # Sometimes there are problems with the servers, wait 10 seconds and try again
                    print('\r' + ' ' * 50 + f'\rSleeping in the "While" because of {e}.... ', end='')
                    sleep(10)
                    print('Done!', end='')

            # Parse the model's output
            thinking_summary = get_gemini_thoughts_summary(output)
            result = output.parsed

            # save the classificatoin
            if result is not None:
                final_labels[key].append(result['classification'])
                text_output[key].append(thinking_summary)
            else:
                # Handle case where the model return nothing...
                print(f"Warning: Failed to parse response for sample {cnt}, skipping...")
                print(output)
                # return None  # ?
                text_output[key].append(thinking_summary)
                final_labels[key].append('failed')

    return final_labels, text_output