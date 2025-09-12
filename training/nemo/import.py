from nemo.collections import llm
from nemo.collections.llm.gpt.model.qwen3 import Qwen3Model, Qwen3Config8B

if __name__ == '__main__':
    llm.import_ckpt(model=Qwen3Model(config=Qwen3Config8B()), source="hf://Qwen/Qwen3-8B-Base")
