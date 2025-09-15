import os
import json
import glob
import random
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm

def count_words(text):
    return len(text.split())

def process_jsonl_file(args):
    file_path, word_to_tokens_ratio = args
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                text = obj.get('text', '').strip()
                if text:
                    num_words = count_words(text)
                    num_tokens = int(num_words * word_to_tokens_ratio)
                    texts.append((text, num_tokens))
            except Exception:
                continue
    return texts

def collect_texts(jsonl_files, word_to_tokens_ratio, token_limit, shuffle=False):
    args = [(f, word_to_tokens_ratio) for f in jsonl_files]
    all_texts = []
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_jsonl_file, args), total=len(args), desc="Reading text"):
            all_texts.extend(result)
    if shuffle:
        random.shuffle(all_texts)
    selected_texts = []
    total_tokens = 0
    for text, num_tokens in all_texts:
        if total_tokens + num_tokens > token_limit:
            break
        selected_texts.append(text)
        total_tokens += num_tokens
    return selected_texts, total_tokens

def main():
    word_to_tokens_ratio = 2.25
    token_limit = int(4.5e9)
    out_path = os.path.join(os.path.dirname(__file__), 'low.jsonl')
    # YifatDataBatch2-Round5
    yifat_dir = os.path.join(os.path.dirname(__file__), 'YifatDataBatch2-Round5')
    yifat_files = glob.glob(os.path.join(yifat_dir, '*.jsonl'))
    yifat_texts, yifat_tokens = collect_texts(yifat_files, word_to_tokens_ratio, token_limit, shuffle=False)
    # English
    english_dir = os.path.join(os.path.dirname(__file__), 'english')
    english_files = glob.glob(os.path.join(english_dir, '*.jsonl'))
    english_texts, english_tokens = collect_texts(english_files, word_to_tokens_ratio, token_limit, shuffle=True)
    # Write output
    with open(out_path, 'w', encoding='utf-8') as out_f:
        for text in tqdm(yifat_texts, desc="Writing Yifat texts"):
            out_f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
        for text in tqdm(english_texts, desc="Writing English texts"):
            out_f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
    print(f"Wrote {len(yifat_texts)} Yifat texts ({yifat_tokens} tokens) and {len(english_texts)} English texts ({english_tokens} tokens) to {out_path}")

if __name__ == "__main__":
    main()
