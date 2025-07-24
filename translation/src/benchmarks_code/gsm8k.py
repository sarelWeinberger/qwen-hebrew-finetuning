from datasets import load_dataset, Dataset


def get_gsm8k_datasets():
    datasets = {}
    # GSM8K 
    # train - 7473 samples | test - 1319 samples
    # datasets['gsm8k_train'] = load_dataset("gsm8k", 'main', split='train')
    datasets['gsm8k_test'] = load_dataset("gsm8k", 'main', split='test')
    return datasets


def gsm8k_sample_to_dict(sample):
    return {
        'question': sample['question'],
        'answer': sample['answer'],
    }


def gsm8k_dict_to_sample(sample, dct):
    sample['question'] = dct['question']
    sample['answer'] = dct['answer']
    return sample

