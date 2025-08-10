import os
import json
from tqdm import tqdm

def count_texts_in_file(filepath):
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                if 'text' in obj:
                    count += 1
            except Exception:
                continue
    return count

def main():
    datasets_dir = '/home/ubuntu/qwen-hebrew-finetuning/qwen_model_shaltiel/datasets'
    results = {}
    for filename in tqdm(os.listdir(datasets_dir)):
        if filename.endswith('.jsonl'):
            path = os.path.join(datasets_dir, filename)
            num_texts = count_texts_in_file(path)
            results[filename] = num_texts
    for fname, count in sorted(results.items()):
        print(f'{fname}: {count}')

if __name__ == '__main__':
    main()
