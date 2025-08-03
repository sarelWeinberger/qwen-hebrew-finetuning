import pandas as pd
import re


def parse_labeled_sample(text):
    """
    This function is brought to you by ChatGPT
    """
    # Extract the example index
    example_match = re.search(r'Example (\d+):', text)
    example_index = int(example_match.group(1)) if example_match else None
    
    # Extract English question and options
    english_block = re.search(r'English:\n(.*?)(?=\nOption 1:)', text, re.DOTALL)
    english_text = english_block.group(1).strip() if english_block else ""
    
    # Extract all option sets
    option1_block = re.search(r'Option 1:\n(.*?)(?=\nOption 2:)', text, re.DOTALL)
    option1_text = option1_block.group(1).strip() if option1_block else ""

    option2_block = re.search(r'Option 2:\n(.*?)(?=\nRating:)', text, re.DOTALL)
    option2_text = option2_block.group(1).strip() if option2_block else ""
    
    # Extract Rating
    rating_match = re.search(r'Rating:(.*)', text)
    rating = rating_match.group(1).strip() if rating_match else ""
    
    # Extract Gold block (if exists)
    gold_block = re.search(r'Gold:\s*(.*)', text, re.DOTALL)
    gold_text = gold_block.group(1).strip() if gold_block else ""
    
    # Assemble the result
    result = {
        'example index': example_index,
        'English': english_text,
        'option 1': option1_text,
        'option 2': option2_text,
        'rating': rating,
        'gold': gold_text
    }
    
    return result


def map_rating_to_model(x, or_df):
    rating = x['rating']
    if x['rating'] not in ['BOTH', 'SKIP', '']:
        rating = or_df.loc[x.name, f"model {x['rating']}"]
    return rating


def split_to_option(x):
    split_in = x['new_text_column'].index('Option 2:')
    a = x['new_text_column'][:split_in]
    b = x['new_text_column'][split_in:]

    # Clean a
    if a.endswith('\n\n'):
        a = a[:-2]
    elif a.endswith('\n    \n'):
        a = a[:-6]
    else:
        pass

    # Clean start string
    if a.startswith('Option 1:\n'):
        a = a[len('Option 1:\n'):]
    
    if b.startswith('Option 2:\n'):
        b = b[len('Option 2:\n'):]
    
    return a, b


map_mqm_to_score = {
    '': 0,
    'minor': 1,
    'major': 5,
    'critical': 25,
}

COLUMNS_1 = [f'severity_annotation_{i}' for i in [1, 2, 3]]
COLUMNS_2 = [f'severity_annotation_{i}' for i in [4, 5, 6]]


def parse_mqm(or_df, label_df):
    models_mqm = {
        k: [] for k in or_df['model 1'].unique()
    }

    score_mqm = {
        k: [] for k in or_df['model 1'].unique()
    }
    
    for indx in or_df.index:
        models_mqm[or_df.loc[indx, 'model 1']] += [
            (
                label_df.loc[indx, f'category_annotation_{i}'],
                label_df.loc[indx, f'severity_annotation_{i}'],
                indx
            ) for i in [1, 2, 3] if label_df.loc[indx, f'category_annotation_{i}'] != ''
        ]
        score_mqm[or_df.loc[indx, 'model 1']] += [
            label_df.loc[indx, COLUMNS_1].apply(lambda x: map_mqm_to_score[x]).sum()
        ]
        
        models_mqm[or_df.loc[indx, 'model 2']] += [
            (
                label_df.loc[indx, f'category_annotation_{i}'],
                label_df.loc[indx, f'severity_annotation_{i}'],
                indx
            ) for i in [4, 5, 6] if label_df.loc[indx, f'category_annotation_{i}'] != ''
        ]
        score_mqm[or_df.loc[indx, 'model 2']] += [
            label_df.loc[indx, COLUMNS_2].apply(lambda x: map_mqm_to_score[x]).sum()
        ]
    return models_mqm, score_mqm


def parse_from_gradio(labeled_file_name, csv_file_name):
    label_df = pd.read_csv(labeled_file_name)
    or_df = pd.read_csv(csv_file_name)

    label_df = label_df.fillna('')
    

    # Split options
    label_df[['option 1', 'option 2']] = label_df.apply(split_to_option, result_type='expand', axis=1)
    
    # Check matching between label_df and or_df
    assert (or_df['original'] == label_df['text_column']).all(), 'problems with original'
    assert (or_df['option 1'] == label_df['option 1']).all(), 'problems with option 1'
    assert (or_df['option 2'] == label_df['option 2']).all(), 'problems with option 2'

    # Add the winning model in each:
    or_df['rating model'] = label_df.apply(lambda x: map_rating_to_model(x, or_df), axis=1)
    or_df['was fixed'] = label_df['gold'] != ''

    return label_df, or_df


def parse_labeled_file(file_name, csv_file_name):
    """
    Args:
        - file_name - a .txt file
    """
    # Read the manual labeled .txt file
    with open(file_name, 'r') as f:
        file_text = f.read()
    # Parse (text parsing)
    # samples = file_text.split('\n' + '-' * 50 + '\n')
    samples = file_text.split('-' * 50)
    # throw the last *always empty* sample
    samples = samples[:-1]
    parsed_samples = [parse_labeled_sample(sample) for sample in samples]

    # To df
    label_df = pd.DataFrame(parsed_samples)

    # Read the csv with the models we used
    or_df = pd.read_csv(csv_file_name)

    # Check matching between label_df and or_df
    assert (or_df['option 1'] == label_df['option 1']).all(), 'problems with option 1'
    assert (or_df['option 2'] == label_df['option 2']).all(), 'problems with option 2'

    label_df = label_df[~label_df['gold'].str.contains('Skip=Yes')]
    or_df = or_df[or_df.index.isin(label_df.index)]

    # Add the winning model in each:
    or_df['rating model'] = label_df.apply(lambda x: map_rating_to_model(x, or_df), axis=1)
    or_df['was fixed'] = label_df['gold'] != ''

    return label_df, or_df
