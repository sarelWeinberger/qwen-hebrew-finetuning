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
    # datasets['mmlu_dev'] = load_dataset("cais/mmlu", "all", split='dev')
    # datasets['mmlu_auxiliary_train'] = load_dataset("cais/mmlu", "all", split='auxiliary_train')
    # datasets['mmlu_val'] = load_dataset("cais/mmlu", "all", split='validation')
    datasets['mmlu_test'] = load_dataset("cais/mmlu", "all", split='test')
    return datasets


def mmlu_sample_to_dict(sample):
    return {
        'question': sample['question'],
        'choice_a': sample['choices'][0],
        'choice_b': sample['choices'][1],
        'choice_c': sample['choices'][2],
        'choice_d': sample['choices'][3],
    }


def mmlu_dict_to_sample(sample, dct):
    sample['question'] = dct['question']
    sample['choices'] = [
        dct['choice_a'],
        dct['choice_b'],
        dct['choice_c'],
        dct['choice_d']
    ]

    # Convert index (0–3) to letter
    index_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    if isinstance(sample['answer'], int):
        letter = index_to_letter[sample['answer']]
    elif str(sample['answer']) in index_to_letter:
        letter = index_to_letter[int(sample['answer'])]
    elif str(sample['answer']) in MAP_LABELS:
        letter = str(sample['answer'])
    else:
        raise ValueError(f"Unexpected answer format: {sample['answer']}")

    sample['answer'] = MAP_LABELS[letter]
    return sample