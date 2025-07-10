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


def get_winogrande_datasets():
    datasets = {}
    # WinoGrande
    # train - 9248 samples | validation - 1267 samples | test - 1767 samples (using debiased version)
    datasets['winogrande_train'] = load_dataset("winogrande", "winogrande_debiased", split='train')
    # datasets['winogrande_val'] = load_dataset("winogrande", "winogrande_debiased", split='validation')
    # datasets['winogrande_test'] = load_dataset("winogrande", "winogrande_debiased", split='test')
    return datasets


def winogrande_sample_to_dict(sample):
    return {
        'question': sample['sentence'],
        'option 1': sample['option1'],
        'option 2': sample['option2'],
    }


def winogrande_dict_to_sample(sample, dct):
    sample['sentence'] = dct['question']
    sample['option1'] = dct['option 1']
    sample['option2'] = dct['option 2']
    # Convert answer (1 or 2) to our label mapping
    if sample['answer'] == '1':
        sample['answer'] = MAP_LABELS['1']
    elif sample['answer'] == '2':
        sample['answer'] = MAP_LABELS['2']
    return sample