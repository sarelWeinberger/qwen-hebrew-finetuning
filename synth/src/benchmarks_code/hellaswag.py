from datasets import load_dataset


def get_hellaswag_datasets():
    datasets = {}
    # HellaSwag
    # train - 39905 samples | validation - 10042 samples | test - 10003 samples
    datasets['hellaswag_train'] = load_dataset("hellaswag", split='train')
    # datasets['hellaswag_val'] = load_dataset("hellaswag", split='validation')
    # datasets['hellaswag_test'] = load_dataset("hellaswag", split='test')
    return datasets


def hellaswag_sample_to_dict(sample):
    return {
        'activity_label': sample['activity_label'],
        'ctx_a': sample['ctx_a'],
        'ctx_b': sample['ctx_b'],
        'ctx': sample['ctx']
    } | {
        f'ending {i}': sample['endings'][i-1] for i in range(1, 5)
    }


def hellaswag_dict_to_sample(sample, dct):
    sample['activity_label'] = dct['activity_label']
    sample['ctx_a'] = dct['ctx_a']
    sample['ctx_b'] = dct['ctx_b']
    sample['ctx'] = dct['ctx']
    sample['endings'] = [dct[f'ending {i}'] for i in range(1, 5)]
    return sample