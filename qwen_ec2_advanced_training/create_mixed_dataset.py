import json
import os
import csv
from pathlib import Path

def count_words(text):
    """Count words in a text string."""
    return len(text.split())

def process_dataset(file_path, max_texts=880, max_words=900, filter_words=True):
    """Process a single dataset file and return filtered texts."""
    texts = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(texts) >= max_texts:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    text = data.get('text', '')
                    
                    if text:
                        # Apply word filter only if filter_words is True
                        if not filter_words or count_words(text) <= max_words:
                            texts.append(text)
                except json.JSONDecodeError:
                    continue
                    
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return texts

def create_mixed_dataset():
    """Create the mixed dataset according to specifications."""
    datasets_dir = "datasets_bkp"
    output_dir = "datasets"
    output_file = os.path.join(output_dir, "mixed_dataset.jsonl")
    order_file = os.path.join(output_dir, "dataset_order.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    if not os.path.exists(datasets_dir):
        print(f"Error: Directory {datasets_dir} does not exist")
        return
    
    # Get all .jsonl files, excluding those starting with "yifat"
    jsonl_files = [f for f in os.listdir(datasets_dir) if f.endswith('.jsonl')]
    jsonl_files = [f for f in jsonl_files if not f.lower().startswith('yifat')]
    jsonl_files = [os.path.join(datasets_dir, f) for f in jsonl_files]

    print(f"Found {len(jsonl_files)} datasets (excluding those starting with 'yifat')")
    
    # Sort files to ensure consistent ordering, but prioritize sefaria and geektime
    ordered_files = []
    
    # First: sefaria (no word filtering)
    sefaria_files = [f for f in jsonl_files if 'sefaria' in os.path.basename(f).lower()]
    if sefaria_files:
        ordered_files.append(sefaria_files[0])
        print(f"Added sefaria: {os.path.basename(sefaria_files[0])}")
    else:
        print("Warning: No sefaria dataset found")
    
    # Second: geektime (with word filtering)
    geektime_files = [f for f in jsonl_files if 'geektime' in os.path.basename(f).lower()]
    if geektime_files:
        ordered_files.append(geektime_files[0])
        print(f"Added geektime: {os.path.basename(geektime_files[0])}")
    else:
        print("Warning: No geektime dataset found")
    
    # Add remaining files (with word filtering)
    used_files = set(os.path.basename(f).lower() for f in ordered_files)
    remaining_files = [f for f in jsonl_files if os.path.basename(f).lower() not in used_files]
    ordered_files.extend(remaining_files)
    
    print(f"\nProcessing {len(ordered_files)} datasets in order:")
    for i, file_path in enumerate(ordered_files, 1):
        print(f"{i}. {os.path.basename(file_path)}")
    
    # Write dataset order to CSV
    with open(order_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Order', 'Dataset_Name', 'File_Path', 'Text_Count', 'Word_Filter_Applied', 'Max_Words'])
        
        # Process datasets and create mixed dataset
        total_texts = 0
        
        with open(output_file, 'w', encoding='utf-8') as output:
            for i, file_path in enumerate(ordered_files, 1):
                dataset_name = os.path.basename(file_path)
                print(f"\nProcessing {dataset_name}...")
                
                # Apply different filtering rules
                if 'sefaria' in dataset_name.lower():
                    # Sefaria: no word filtering
                    texts = process_dataset(file_path, max_texts=880, filter_words=False)
                    word_filter = "No"
                    max_words = "N/A"
                else:
                    # All others: word filtering applied
                    texts = process_dataset(file_path, max_texts=880, max_words=900, filter_words=True)
                    word_filter = "Yes"
                    max_words = "900"
                
                print(f"Found {len(texts)} valid texts")
                
                # Write to CSV
                writer.writerow([i, dataset_name, file_path, len(texts), word_filter, max_words])
                
                # Write texts to output file
                for text in texts:
                    json_line = json.dumps({"text": text}, ensure_ascii=False)
                    output.write(json_line + '\n')
                    total_texts += 1
    
    print(f"\nMixed dataset created: {output_file}")
    print(f"Dataset order saved: {order_file}")
    print(f"Total texts: {total_texts}")
    print(f"Expected texts: {len(ordered_files) * 880}")

if __name__ == "__main__":
    create_mixed_dataset()