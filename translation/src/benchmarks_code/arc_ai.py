from datasets import load_dataset, Dataset

MAP_LABELS = {
    'A': 'א',
    'B': 'ב',
    'C': 'ג',
    'D': 'ד',
    'E': 'ה',
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
}


def get_arc_ai2_datasets():
    datasets = {}

    # ARC_AI2 - Challenge
    # train - 1119 samples | validation - 299 samples | test - 1172 samples
    datasets['arc_challenge_train'] = load_dataset("allenai/ai2_arc", 'ARC-Challenge', split='train') 
    # datasets['arc_challenge_val'] = load_dataset("allenai/ai2_arc", 'ARC-Challenge', split='validation')
    # datasets['arc_challenge_test'] = load_dataset("allenai/ai2_arc", 'ARC-Challenge', split='test')
    
    # ARC_AI2 - Easy
    # train - 2251 samples | validation - 570 samples | test - 2376 samples
    # datasets['arc_easy_train'] = load_dataset("allenai/ai2_arc", 'ARC-Easy', split='train')
    # datasets['arc_easy_val'] = load_dataset("allenai/ai2_arc", 'ARC-Easy', split='validation')
    # datasets['arc_easy_test'] = load_dataset("allenai/ai2_arc", 'ARC-Easy', split='test')
    return datasets


def arc_sample_to_dict(sample):
    return {
        'question': sample['question'],
        'option 1': sample['choices']['text'][0],
        'option 2': sample['choices']['text'][1],
        'option 3': sample['choices']['text'][2],
        'option 4': sample['choices']['text'][3],
    }


def arc_dict_to_sample(sample, dct):
    sample['question'] = dct['question']
    sample['choices']['text'] = [dct[f'option {i}'] for i in range(1, 5)]
    sample['choices']['label'] = [MAP_LABELS[label] for label in sample['choices']['label']]
    sample['answerKey'] = MAP_LABELS[sample['answerKey']]
    return sample
    

# def into_arc_prompt_v2(id, question, choices, answerKey):
#     answers_str = '[' + ']\n['.join(choices['text']) + ']'
#     return f"<question>{question}</question>\n<answers>\n{answers_str}\n</answers>"


# def back_to_arc_v2(x, text):
#     # x - a sample from the dataset
#     question_match = re.search(r"<question>(.*?)</question>", text, re.DOTALL)
#     question = question_match.group(1).strip()
    
#     # Extract all answers within square brackets
#     answers_match = re.findall(r"\[(.*?)\]", text, re.DOTALL)
#     answers = [ans.strip() for ans in answers_match]

#     x['question'] = question
#     x['choices']['text'] = answers
#     x['choices']['label'] = [MAP_LABELS[label] for label in x['choices']['label']]
#     x['answerKey'] = MAP_LABELS[x['answerKey']]
#     return x