import os
import json
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm
import random


def basic_text_clean(text):
    # Normalize unicode (especially useful for cases where the menukad texts use characters
    # like U+FB2C = shin+dagesh+shin-dot in one unicode char. This splits them into regular Hebrew. 
    text = unicodedata.normalize('NFKC', text)
    
    # Normalize line endings to only \r, no \n
    text = text.replace('\r\n', '\n').replace('\r', '\n').strip()
    
    # weird char cleaning (zero-width and other problematic characters) + nbsp
    text = re.sub('&nbsp;?', ' ', text)
    text = re.sub(r'[\u200B-\u200F\u202A-\u202C\x00\f\u0013-\u0015\u0005\u0090]', '', text)
    text = text.replace('\u00A0', ' ')
    
    # Remove leading/trailing whitespace (non-line-break) from each line (except newlines)
    text = re.sub(r'^[^\S\n]*|[^\S\n]*$', '', text, flags=re.MULTILINE)

    # Fix double line/breaks spaces
    text = re.sub(r'\n\n+', '\n\n', text)
    text = re.sub(r'  +', ' ', text)
    
    # Remove nikud
    text = re.sub(r'[\u0597-\u05C7]', '', text)

    return text

def process_ben_yehuda():
    """Process Ben Yehuda text files and create JSONL output."""
    input_dir = r"./public_domain_dump/txt"
    output_file = r"ben_yehuda_no_nikud_mar25dump.jsonl"
    
    with open(output_file, 'w', encoding='utf-8', newline='\n') as outfile:
        # Get all files recursively
        all_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                all_files.append(os.path.join(root, file))
        
        for file_path in tqdm(all_files, desc="Processing files"):
            text_lines = []
            prev_was_empty = False
            
            file_key = file_path[file_path.index('public_domain_dump'):].replace('\\', '/')

            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    line = line.strip()
                    
                    # Skip lines containing "פרויקט בן־יהודה"
                    if "פרויקט בן־יהודה" in line:
                        continue
                    
                    # logic to handle real line-breaks vs fake (denominated by two line-breaks)
                    if len(line) == 0:
                        if prev_was_empty:
                            text_lines.append('')
                        prev_was_empty = True
                    else:
                        text_lines.append(line)
                        prev_was_empty = False
            
            # Join all lines and clean the text
            full_text = '\n'.join(text_lines)
            cleaned_text = basic_text_clean(full_text)
            
            # Write as JSON line
            json_line = json.dumps(dict(text=cleaned_text, path=file_key), ensure_ascii=False)
            outfile.write(json_line + '\n')
            
    print(f"Processing complete. Output saved to: {output_file}")

if __name__ == "__main__":
    process_ben_yehuda()