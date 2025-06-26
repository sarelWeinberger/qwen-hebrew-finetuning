from src.call_models import call_claude_bedrock
from src.call_models import call_gemini, all_string_gemini_config
from datasets import Dataset
import re
from tqdm.notebook import tqdm

# The base prompt of the English-Hebrew
BASE_PROMPT = "English:\n{X}\nHebrew:\n{Y}"


def claude_translation(bedrock_client, datasets, instruct, few_shots, sample_format, sample_to_dict, dict_to_sample):
    """
    Translate all given datasets using one of the claude famaliy models.
    """
    # The final prompt for the models
    final_prompt = instruct + '\n\n' + few_shots + '\n\n' + sample_format + '\n\n'
    hebrew_dataset = {}
    # Run on the different splits in the dataset
    for key in datasets:
        print(f'Translating {key}...')
        hebrew_dataset[key] = []
        # Run on all the split's samples
        for sample in tqdm(datasets[key], total=datasets[key].num_rows):
            # from sample to dict
            dct = sample_to_dict(sample)

            # Enter into prompt
            samples_query = '\n'.join([f'<{k}>{dct[k]}</{k}>' for k in dct])
            query = final_prompt + BASE_PROMPT.format(X=samples_query, Y='')

            # Call claude
            output = call_claude_bedrock(bedrock_client, query)

            # Parse the model's output
            pattern = r"<([^>]+)>(.*?)</\1>"
            matches = re.findall(pattern, output, re.DOTALL)
            result = {key: value.strip() for key, value in matches}

            # Create New sample
            new_sample = dict_to_sample(sample, result)
            hebrew_dataset[key].append(new_sample)

        hebrew_dataset[key] = Dataset.from_list(hebrew_dataset[key])
    return hebrew_dataset


def gemini_translation(google_client, datasets, instruct, few_shots, sample_to_dict, dict_to_sample):
    """
    Translate all given datasets using one of the Gemini famaliy models.
    """
    # The final prompt for the models
    final_prompt = few_shots + '\n\n'
    hebrew_dataset = {}
    # Run on the different splits in the dataset
    for key in datasets:
        fields = sample_to_dict(datasets[key][0]).keys()
        config = all_string_gemini_config(fields, instruct)

        print(f'Translating {key}...')
        hebrew_dataset[key] = []
        # Run on all the split's samples
        for sample in tqdm(datasets[key], total=datasets[key].num_rows):
            # from sample to dict
            dct = sample_to_dict(sample)

            # Enter into prompt
            samples_query = '\n'.join([f'<{k}>{dct[k]}</{k}>' for k in dct])
            query = final_prompt + BASE_PROMPT.format(X=samples_query, Y='')

            # Call gemini
            output = call_gemini(google_client, query, config)

            # Parse the model's output
            result = output.parsed

            # Create New sample
            new_sample = dict_to_sample(sample, result)
            hebrew_dataset[key].append(new_sample)

        hebrew_dataset[key] = Dataset.from_list(hebrew_dataset[key])
    return hebrew_dataset