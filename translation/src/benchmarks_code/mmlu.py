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

def get_mmlu_datasets():
    datasets = {}
    # MMLU
    # dev - 285 samples | val - 1531 samples | test - 14042 samples
    datasets['mmlu_dev'] = load_dataset("cais/mmlu", "all", split='dev')
    # datasets['mmlu_val'] = load_dataset("cais/mmlu", "all", split='validation')
    # datasets['mmlu_test'] = load_dataset("cais/mmlu", "all", split='test')
    return datasets

def mmlu_sample_to_dict(sample):
    return {
        'question': sample['question'],
        'option 1': sample['choices'][0],
        'option 2': sample['choices'][1],
        'option 3': sample['choices'][2],
        'option 4': sample['choices'][3],
    }

def mmlu_dict_to_sample(sample, dct):
    sample['question'] = dct['question']
    sample['choices'] = [dct[f'option {i}'] for i in range(1, 5)]
    sample['answer'] = MAP_LABELS[str(sample['answer'])]
    return sample