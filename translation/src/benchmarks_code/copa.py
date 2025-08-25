from datasets import load_dataset

MAP_TYPE = {
    'cause': 'סיבה',
    'effect': 'תוצאה',
}


def get_copa_datasets():
    datasets = {}
    # ARC_AI2 - Challenge
    # train - 500 samples | test - 500 samples
    datasets['copa_train'] = load_dataset("pkavumba/balanced-copa", split='train') 
    datasets['copa_test'] = load_dataset("pkavumba/balanced-copa", split='test')

    return datasets


def copa_sample_to_dict(sample):
    return {
        'premise': sample['premise'],
        'choice1': sample['choice1'],
        'choice2': sample['choice2'],
    }


def copa_dict_to_sample(sample, dct):
    sample['premise'] = dct['premise']
    sample['choice1'] = dct['choice1']
    sample['choice2'] = dct['choice2']
    sample['question'] = MAP_TYPE[sample['question']]
    return sample
