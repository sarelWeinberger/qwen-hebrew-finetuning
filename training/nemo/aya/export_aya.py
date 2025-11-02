from aya import *
from nemo.collections import llm

if __name__ == '__main__':
    llm.export_ckpt('/path/to/nemo/checkpoint', "hf", '/path/to/save/hf/', overwrite=True)