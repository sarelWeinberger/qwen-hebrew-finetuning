from nemo.collections import llm
from aya import *

if __name__ == '__main__':
    llm.import_ckpt(model=CohereModel(config=AyaExpanseConfig8B()), source="hf://CohereLabs/aya-expanse-8b")
    llm.import_ckpt(model=CohereModel(config=AyaExpanseConfig32B()), source="hf://CohereLabs/aya-expanse-32b")