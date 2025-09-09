from datasets import load_dataset


def get_alpaca_datasets():
    datasets = {}

    # Alpaca dataset - ~52k samples
    return load_dataset("yahma/alpaca-cleaned", split='train', streaming=True)
