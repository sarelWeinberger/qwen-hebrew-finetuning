import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


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


def parse_single_file_gradio(labeled_file_name):
    label_df = pd.read_csv(labeled_file_name)
    label_df = label_df.fillna('')

    # Split options
    label_df[['option 1', 'option 2']] = label_df.apply(split_to_option, result_type='expand', axis=1)

    # label_df['rating model'] = label_df.apply(lambda x: map_rating_to_model(x, label_df), axis=1)
    label_df['rating model'] = label_df.apply(lambda x: x[f'model {x["rating"]}'] if x['rating'] in ['1', '2'] else x['rating'], axis=1)
    label_df['was fixed'] = label_df['gold'] != ''

    label_df.rename({
        'text_column': 'original',
    }, axis=1, inplace=True)

    return label_df


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

    or_df['original'] = or_df['original'].apply(lambda x: x.replace('\xa0', ' '))
    or_df['option 1'] = or_df['option 1'].apply(lambda x: x.replace('\xa0', ' '))
    or_df['option 2'] = or_df['option 2'].apply(lambda x: x.replace('\xa0', ' '))

    # return label_df, or_df

    # Check matching between label_df and or_df
    assert (or_df['option 1'] == label_df['option 1']).all(), 'problems with option 1'
    assert (or_df['option 2'] == label_df['option 2']).all(), 'problems with option 2'

    label_df = label_df[~label_df['gold'].str.contains('Skip=Yes')]
    or_df = or_df[or_df.index.isin(label_df.index)]

    # Add the winning model in each:
    or_df['rating model'] = label_df.apply(lambda x: map_rating_to_model(x, or_df), axis=1)
    or_df['was fixed'] = label_df['gold'] != ''

    return label_df, or_df


def rating_gold_metrics(or_df, bench_name='', annotate=True, ax=None):
    show = False
    if ax is None:
        _, ax = plt.subplots(1, 1)
        show = True

    # Clauclate number of choosing each model - and how many times was it fixed to 'gold'
    rating = or_df[or_df['rating model'] != 'SKIP']
    group_1 = rating[~rating['was fixed']]['rating model'].value_counts()
    group_2 = rating[rating['was fixed']]['rating model'].value_counts()

    # Add values for missing indecies if needed:
    all_models = group_1.index.union(group_2.index)  # Get all unique model names
    group_1 = group_1.reindex(all_models, fill_value=0)
    group_2 = group_2.reindex(all_models, fill_value=0)

    # stacked bat plot
    ax.bar(group_1.index, group_1.values, label='Already gold')
    ax.bar(group_2.index, group_2.values, bottom=group_1, label='Was fixed')

    # Add annotations
    for i, model in enumerate(group_1.index):
        y1 = group_1.values[i]
        y2 = group_2.values[i]

        if annotate:
            if y1 > 0:
                ax.text(i, y1 / 2, f'{np.round(100 * y1 / (y1 + y2), 2)}%', ha='center', va='center', color='white', fontsize=12)
            if y2 > 0:
                ax.text(i, y1 + y2 / 2, f'{np.round(100 * y2 / (y1 + y2), 2)}%', ha='center', va='center', color='white', fontsize=12)

        ax.text(i, y1 + y2, f'{y1 + y2}', ha='center', color='black', fontsize=12)
    ax.legend()
    ax.set_ylim(0, (group_1.max() + group_2.max()) * 1.05)
    ax.set_xlabel('Choosen model')
    ax.set_ylabel('Count')
    ax.set_title(f'{bench_name} Better model comparison')
    if show:
        plt.show()


categories_x = [
    "Adequacy - Mistranslation",
    "Adequacy - Omission",
    "Adequacy - Addition",
    "Adequacy - TerminologyNamedEntity",
    "Adequacy - CulturalReference",
    "Fluency - Agreement",
    "Fluency - MorphologyFunction",
    "Fluency - WordOrderSyntax",
    "Fluency - OrthographyPunct",
    "LocaleStyle - Register",
    "LocaleStyle - Conventions",
]

severity_x = [
    'minor',
    'major',
    'critical',
]


def MQM_metrics(mqm_res, mqm_score, bench_name='', files_name=''):
    print('MQM average scores:')
    for k in mqm_score:
        print(f'\t{k:20} - {sum(mqm_score[k]) / len(mqm_score[k])}')
    print()
    corr = pd.DataFrame(mqm_score).corr().values[0, 1]
    print(f'\tThere is a correlation in the MQM scores of: {corr:.3}')

    # ----- MQM scores histogram
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    for k in mqm_score:
        # pd.Series(mqm_score[k]).value_counts().plot(alpha=0.5, label=k, kind='bar')
        axs[0].hist(mqm_score[k], alpha=0.5, label=k, range=(-0.5, 8.5), bins=9)
    axs[0].legend()
    axs[0].grid()
    axs[0].set_xticks(range(9))
    axs[0].set_xlabel('MQM score', fontsize=11)
    axs[0].set_ylabel('Count', fontsize=11)
    axs[0].set_title('Below 10', fontsize=13)

    for k in mqm_score:
        # pd.Series(mqm_score[k]).value_counts().plot(alpha=0.5, label=k, kind='bar')
        axs[1].hist(mqm_score[k], alpha=0.5, label=k, range=(24.5, 30.5), bins=7)
    axs[1].legend()
    axs[1].grid()
    axs[1].set_xticks(range(25, 32))
    axs[1].set_xlabel('MQM score', fontsize=11)
    axs[1].set_ylabel('Count', fontsize=11)
    axs[1].set_title('Above 25', fontsize=13)
    
    fig.suptitle(f'{bench_name} MQM scores histogram', fontsize=15)
    fig.subplots_adjust(hspace=0.6)
    fig.tight_layout()
    plt.savefig(f'plots/{files_name}_mqm_hist.jpeg')
    plt.show()

    print('\n' + '-' * 30, end='\n')
    # ----- MQM severity distribution
    fig, axs = plt.subplots(len(mqm_res), 1, figsize=(6, 6), sharex=True)
    for i, k in enumerate(mqm_res):
        severities = pd.DataFrame(mqm_res[k])[1].value_counts(normalize=False)
        severities = severities.reindex(severity_x, fill_value=0).loc[severity_x]
        axs[i].bar(severities.index, severities.values, alpha=0.8)
        axs[i].grid()
        axs[i].set_title(k)
        # axs[i].set_xlabel('MQM severity')
        # axs[i].set_yticks(np.linspace(0, 1, 11))

        # Add annotations
        for place, y1 in enumerate(severities.values):
            axs[i].text(place, y1, f'{np.round(100 * y1 / severities.sum(), 2)}%', ha='center', color='black', fontsize=12)
    
    # fig.supxlabel('MQM severity')
    fig.subplots_adjust(hspace=0.6)
    # plt.ylabel('Count')
    fig.suptitle(f'{bench_name} MQM severity distribution')
    fig.tight_layout()
    plt.savefig(f'plots/{files_name}_mqm_sev_dist.jpeg')
    plt.show()

    print('\n' + '-' * 30, end='\n')
    # ----- MQM category distribution
    fig, axs = plt.subplots(1 + len(mqm_res), 1, figsize=(6, 12), sharex=True)
    
    for k in mqm_res:
        categories = pd.DataFrame(mqm_res[k])[0].value_counts(normalize=True)
        categories = categories.reindex(categories_x, fill_value=0)
        categories = categories.loc[categories_x]
        axs[0].bar(categories.index, categories.values, alpha=0.5, label=k)
    axs[0].legend()
    axs[0].grid()
    # axs[0].set_xticks(rotation=90)
    # axs[0].set_xlabel('MQM category')
    # plt.ylabel('Count')
    axs[0].set_title('distribution')

    # MQM minor-major distribution
    for indx, k in enumerate(mqm_res, start=1):
        group_1 = [mqm_res[k][i][0] for i in range(len(mqm_res[k])) if mqm_res[k][i][1] == 'minor']
        group_2 = [mqm_res[k][i][0] for i in range(len(mqm_res[k])) if mqm_res[k][i][1] == 'major']
        group_3 = [mqm_res[k][i][0] for i in range(len(mqm_res[k])) if mqm_res[k][i][1] == 'critical']

        group_1 = pd.Series(group_1).value_counts()
        group_2 = pd.Series(group_2).value_counts()
        group_3 = pd.Series(group_3).value_counts()
    
        # Add values for missing indecies if needed:
        group_1 = group_1.reindex(categories_x, fill_value=0).loc[categories_x]
        group_2 = group_2.reindex(categories_x, fill_value=0).loc[categories_x]
        group_3 = group_3.reindex(categories_x, fill_value=0).loc[categories_x]
    
        # stacked bat plot
        axs[indx].bar(group_1.index, group_1.values, label='minor')
        axs[indx].bar(group_2.index, group_2.values, bottom=group_1, label='major')
        X = group_1 + group_2
        axs[indx].bar(group_3.index, group_3.values, bottom=X, label='critical')
    
        # Add annotations
        for i, _ in enumerate(categories_x):
            y1 = group_1.values[i]
            y2 = group_2.values[i]
            y3 = group_3.values[i]
            
            if y1 > 0:
                axs[indx].text(i, y1 / 2, f'{y1 / (y1 + y2 + y3):.3}', ha='center', va='center', color='white', fontsize=8)
            if y2 > 0:
                axs[indx].text(i, y1 + y2 / 2 - 2 * (y3 > 0), f'{y2 / (y1 + y2 + y3):.3}', ha='center', va='center', color='white', fontsize=8)
            if y3 > 0:
                axs[indx].text(i, y1 + y2 + y3 / 2, f'{y3 / (y1 + y2 + y3):.3}', ha='center', va='center', color='white', fontsize=8)
    
        axs[indx].legend()
        axs[indx].set_ylim(0, (group_1.max() + group_2.max() + group_3.max()) * 1.1)
        axs[indx].grid()
        # axs[indx].set_xlabel('MQM category')
        # axs[indx].xticks(rotation=90)
        axs[indx].set_title(f'{k} - histogram')

    plt.setp(axs[2].get_xticklabels(), rotation=90, ha="center")
    
    fig.supxlabel('MQM category')
    fig.subplots_adjust(hspace=0.6)
    # plt.ylabel('Count')
    fig.suptitle(f'{bench_name} MQM category')
    fig.tight_layout()
    plt.savefig(f'plots/{files_name}_mqm_cat_dist.jpeg')
    plt.show()
