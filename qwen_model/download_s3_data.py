import os
import argparse
import boto3
import json
from tqdm import tqdm
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Download and process JSONL data from S3 for fine-tuning")
    parser.add_argument(
        "--aws_access_key_id",
        type=str,
        help="AWS Access Key ID"
    )
    parser.add_argument(
        "--aws_secret_access_key",
        type=str,
        help="AWS Secret Access Key"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="gepeta-datasets",
        help="S3 bucket name"
    )
    parser.add_argument(
        "--key",
        type=str,
        default="processed/wikipedia",
        help="S3 key/path prefix"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="qwen_model/data",
        help="Directory to save downloaded and processed data"
    )
    return parser.parse_args()

def configure_aws(args):
    """Configure AWS credentials"""
    if args.aws_access_key_id and args.aws_secret_access_key:
        # Use provided credentials
        print("Using provided AWS credentials...")
        # Create the session with the credentials
        session = boto3.Session(
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
            region_name=args.region
        )
        # Return the session for creating clients
        return session
    else:
        # Try to use default credentials
        print("No AWS credentials provided. Trying to use default credentials...")
        session = boto3.Session(region_name=args.region)
        return session

def download_s3_files(args, session):
    """Download JSONL files from S3"""
    # Create S3 client using the session
    s3 = session.client('s3')
    
    # Create output directory
    raw_data_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # Track if any downloads were successful
    download_success = False
    
    # First, test the connection and permissions
    try:
        print("Testing S3 connection and permissions...")
        s3.head_bucket(Bucket=args.bucket)
        print(f"Successfully connected to bucket: {args.bucket}")
    except Exception as e:
        print(f"Error connecting to S3 bucket {args.bucket}: {e}")
        print("Will create sample data instead.")
        return create_sample_data(raw_data_dir)
    
    # List and download all JSONL files from the specified key/path
    try:
        print(f"Listing all JSONL files in {args.bucket}/{args.key}...")
        
        # Use paginator to handle large numbers of files
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=args.bucket,
            Prefix=args.key
        )
        
        jsonl_files_found = 0
        
        for page in page_iterator:
            if 'Contents' in page:
                for item in page['Contents']:
                    s3_key = item['Key']
                    filename = os.path.basename(s3_key)
                    
                    # Only process JSONL files
                    if not filename.lower().endswith('.jsonl'):
                        continue
                    
                    jsonl_files_found += 1
                    local_path = os.path.join(raw_data_dir, filename)
                    
                    # Skip if already downloaded
                    if os.path.exists(local_path):
                        print(f"File {filename} already exists. Skipping download.")
                        download_success = True
                        continue
                    
                    print(f"Downloading {s3_key} to {local_path}...")
                    try:
                        s3.download_file(args.bucket, s3_key, local_path)
                        print(f"Downloaded {filename} successfully.")
                        download_success = True
                    except Exception as e:
                        print(f"Error downloading {s3_key}: {e}")
        
        if jsonl_files_found == 0:
            print(f"No JSONL files found in {args.bucket}/{args.key}")
        else:
            print(f"Found {jsonl_files_found} JSONL files in total.")
            
    except Exception as e:
        print(f"Error listing JSONL files: {e}")
    
    # If no downloads were successful, create sample data
    if not download_success:
        print("No files were successfully downloaded. Creating sample data instead.")
        return create_sample_data(raw_data_dir)
    
    return raw_data_dir

def create_sample_data(raw_data_dir):
    """Create sample Hebrew Wikipedia data for testing when S3 download fails"""
    print("Creating sample Hebrew Wikipedia data for testing...")
    
    # Sample Hebrew Wikipedia-style text data
    sample_data = [
        {
            "text": "ויקיפדיה היא אנציקלופדיה רב-לשונית, חופשית ופתוחה לעריכה, הנכתבת בשיתוף על ידי מתנדבים ברחבי העולם וכמעט כל אחד יכול לערוך בה."
        },
        {
            "text": "ישראל, רשמית מדינת ישראל, היא מדינה במזרח התיכון השוכנת על החוף המזרחי של הים התיכון. ישראל גובלת בלבנון בצפון, בסוריה וירדן במזרח, במצרים בדרום מערב."
        },
        {
            "text": "תל אביב-יפו היא עיר במחוז תל אביב בישראל ובירתו הכלכלית של המדינה. העיר נוסדה בשנת 1909 על ידי מייסדי אחוזת בית כשכונת גנים יהודית על חולות צפונית ליפו."
        },
        {
            "text": "ירושלים היא בירת ישראל ועיר הבירה הגדולה ביותר במדינה מבחינת אוכלוסייה ושטח. ירושלים שוכנת בהרי יהודה, בין ים המלח במזרח לים התיכון במערב."
        },
        {
            "text": "השפה העברית היא שפה שמית ממשפחת השפות האפרו-אסיאתיות. העברית היא השפה הרשמית העיקרית בישראל לצד הערבית, ומשמשת כשפת הדיבור של רוב אזרחי ישראל."
        },
        {
            "text": "האוניברסיטה העברית בירושלים היא מוסד להשכלה גבוהה הממוקם בירושלים. האוניברסיטה נוסדה בשנת 1918 והיא מהאוניברסיטאות הוותיקות והמובילות בישראל."
        },
        {
            "text": "ים המלח הוא אגם מלח הממוקם בבקעת ירדן, בין ישראל וירדן. זהו הנקודה הנמוכה ביותר על פני כדור הארץ, כ-430 מטר מתחת לפני הים."
        },
        {
            "text": "מדינת ישראל הוקמה ב-14 במאי 1948, עם תום המנדט הבריטי על פלשתינה, על פי החלטת האומות המאוחדות מנובמבר 1947 על חלוקת פלשתינה."
        }
    ]
    
    # Create a sample JSONL file
    sample_file_path = os.path.join(raw_data_dir, "sample_wikipedia_data.jsonl")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(sample_file_path), exist_ok=True)
    
    # Write sample data to JSONL
    with open(sample_file_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Created sample data file at {sample_file_path}")
    return raw_data_dir

def process_jsonl_files(raw_data_dir, output_dir):
    """Process downloaded JSONL files and prepare for fine-tuning"""
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Get all JSONL files
    jsonl_files = glob.glob(os.path.join(raw_data_dir, "*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files to process.")
    
    # Process each file
    all_texts = []
    
    for jsonl_file in jsonl_files:
        filename = os.path.basename(jsonl_file)
        print(f"Processing {filename}...")
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                line_count = 0
                valid_texts = 0
                
                for line in f:
                    line_count += 1
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Parse JSON line
                        data = json.loads(line)
                        
                        # Extract text - try common field names
                        text = None
                        for possible_field in ['text', 'content', 'article', 'body', 'document']:
                            if possible_field in data and data[possible_field]:
                                text = data[possible_field]
                                break
                        
                        if text and isinstance(text, str) and len(text.strip()) > 50:
                            all_texts.append(text.strip())
                            valid_texts += 1
                    
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_count} in {filename}: {e}")
                        continue
                    except Exception as e:
                        print(f"Warning: Error processing line {line_count} in {filename}: {e}")
                        continue
                
                print(f"Extracted {valid_texts} valid texts from {line_count} lines in {filename}.")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"Total extracted texts: {len(all_texts)}")
    
    # If no texts were extracted, create a minimal dataset
    if not all_texts:
        print("No texts were extracted from JSONL files. Creating minimal sample dataset.")
        all_texts = [
            "ויקיפדיה היא אנציקלופדיה רב-לשונית, חופשית ופתוחה לעריכה, הנכתבת בשיתוף על ידי מתנדבים ברחבי העולם.",
            "ישראל, רשמית מדינת ישראל, היא מדינה במזרח התיכון השוכנת על החוף המזרחי של הים התיכון.",
            "תל אביב-יפו היא עיר במחוז תל אביב בישראל ובירתו הכלכלית של המדינה.",
            "ירושלים היא בירת ישראל ועיר הבירה הגדולה ביותר במדינה מבחינת אוכלוסייה ושטח.",
            "השפה העברית היא שפה שמית ממשפחת השפות האפרו-אסיאתיות."
        ]
    
    # Save as JSONL for fine-tuning
    output_file = os.path.join(processed_dir, "wikipedia_dataset.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in tqdm(all_texts, desc="Saving texts"):
            if text and len(text.strip()) > 50:
                f.write(json.dumps({"text": text.strip()}, ensure_ascii=False) + '\n')
    
    print(f"Processed data saved to {output_file}")
    return output_file

def prepare_for_training(jsonl_file, output_dir):
    """Prepare the processed data for training using the prepare_dataset.py script"""
    dataset_dir = os.path.join(output_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create a command to run the prepare_dataset.py script
    cmd = f"python qwen_model/prepare_dataset.py --input_file {jsonl_file} --output_dir {dataset_dir}"
    print(f"Running command: {cmd}")
    
    # Run the command
    result = os.system(cmd)
    
    if result == 0:
        print(f"Dataset prepared for training and saved to {dataset_dir}")
        print("\nTo use this dataset for fine-tuning, run:")
        print(f"python qwen_model/train.py --dataset_path {dataset_dir}/dataset --config qwen_model/finetuning/training_config.json")
    else:
        print("Error occurred while preparing dataset. Please check the prepare_dataset.py script.")

def main():
    args = parse_args()
    
    print("\n=== Downloading and Processing Wikipedia JSONL Data for Fine-tuning ===\n")
    print(f"Target: s3://{args.bucket}/{args.key}")
    
    # Configure AWS and get session
    session = configure_aws(args)
    
    # Download files from S3
    raw_data_dir = download_s3_files(args, session)
    
    # Process JSONL files
    jsonl_file = process_jsonl_files(raw_data_dir, args.output_dir)
    
    # Prepare for training
    prepare_for_training(jsonl_file, args.output_dir)

if __name__ == "__main__":
    main()