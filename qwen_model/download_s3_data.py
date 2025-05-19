import os
import argparse
import boto3
import pandas as pd
import json
from tqdm import tqdm
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Download and process data from S3 for fine-tuning")
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
        default="israllm-datasets",
        help="S3 bucket name"
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
    """Download specified files from S3"""
    # Create S3 client using the session
    s3 = session.client('s3')
    
    # Create output directory
    raw_data_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # Specific files to download
    specific_files = [
        "csv-dataset/AllOfHEOscarData-Combined-Deduped-DC4_part001.csv",
        "csv-dataset/AllOfNewHebrewWikipediaWithArticles-Oct29-2023_part020.csv",
        "csv-dataset/AllTzenzuraData-Combined-Deduped-DC4_part024.csv"
    ]
    
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
    
    # Download specific files
    for s3_key in specific_files:
        filename = os.path.basename(s3_key)
        local_path = os.path.join(raw_data_dir, filename)
        
        if os.path.exists(local_path):
            print(f"File {filename} already exists. Skipping download.")
            download_success = True
            continue
        
        print(f"Downloading {s3_key} to {local_path}...")
        try:
            # First check if the file exists
            s3.head_object(Bucket=args.bucket, Key=s3_key)
            # Then download it
            s3.download_file(args.bucket, s3_key, local_path)
            print(f"Downloaded {filename} successfully.")
            download_success = True
        except Exception as e:
            print(f"Error downloading {s3_key}: {e}")
    
    # List and download all files starting with AllTzenzuraData
    try:
        print("Listing all files starting with AllTzenzuraData...")
        response = s3.list_objects_v2(
            Bucket=args.bucket,
            Prefix="csv-dataset/AllTzenzuraData"
        )
        
        if 'Contents' in response:
            for item in response['Contents']:
                s3_key = item['Key']
                filename = os.path.basename(s3_key)
                local_path = os.path.join(raw_data_dir, filename)
                
                # Skip if already downloaded
                if os.path.exists(local_path):
                    print(f"File {filename} already exists. Skipping download.")
                    download_success = True
                    continue
                
                # Skip if it's one of the specific files we already downloaded
                if s3_key in specific_files:
                    continue
                
                print(f"Downloading {s3_key} to {local_path}...")
                try:
                    s3.download_file(args.bucket, s3_key, local_path)
                    print(f"Downloaded {filename} successfully.")
                    download_success = True
                except Exception as e:
                    print(f"Error downloading {s3_key}: {e}")
    except Exception as e:
        print(f"Error listing AllTzenzuraData files: {e}")
    
    # If no downloads were successful, create sample data
    if not download_success:
        print("No files were successfully downloaded. Creating sample data instead.")
        return create_sample_data(raw_data_dir)
    
    return raw_data_dir

def create_sample_data(raw_data_dir):
    """Create sample Hebrew data for testing when S3 download fails"""
    print("Creating sample Hebrew data for testing...")
    
    # Sample Hebrew text data
    sample_data = [
        {
            "text": "שלום עולם! זוהי דוגמה לטקסט בעברית לצורך אימון מודל השפה. אנחנו מנסים ליצור דוגמאות שיעזרו למודל ללמוד את השפה העברית."
        },
        {
            "text": "עברית היא שפה שמית, ממשפחת השפות האפרו-אסיאתיות, הנהוגה כשפה רשמית במדינת ישראל. העברית היא השפה הרשמית העיקרית בישראל, ומשמשת כשפת הדיבור של רוב אזרחי ישראל."
        },
        {
            "text": "תל אביב היא עיר במחוז תל אביב בישראל, ובירתו הכלכלית של המדינה. העיר תל אביב נוסדה בשנת 1909 כשכונת גנים יהודית על חולות צפונית ליפו, והיא חלק מגוש דן."
        },
        {
            "text": "ירושלים היא בירת ישראל ועיר הבירה הגדולה ביותר במדינה. ירושלים שוכנת בהרי יהודה, בין ים המלח לים התיכון. העיר העתיקה של ירושלים וחומותיה הן אתר מורשת עולמית של אונסק\"ו."
        },
        {
            "text": "השפה העברית התפתחה לאורך אלפי שנים, מהעברית המקראית, דרך לשון חז\"ל, ימי הביניים, ועד לתחייתה בעת החדשה על ידי אליעזר בן-יהודה ואחרים."
        }
    ]
    
    # Create a sample CSV file
    sample_file_path = os.path.join(raw_data_dir, "sample_hebrew_data.csv")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(sample_file_path), exist_ok=True)
    
    # Write sample data to CSV
    with open(sample_file_path, 'w', encoding='utf-8') as f:
        f.write("text\n")  # Header
        for item in sample_data:
            # Escape any commas in the text and wrap in quotes to ensure proper CSV format
            escaped_text = f"\"{item['text'].replace('\"', '\"\"')}\""
            f.write(f"{escaped_text}\n")
    
    print(f"Created sample data file at {sample_file_path}")
    return raw_data_dir

def process_csv_files(raw_data_dir, output_dir):
    """Process downloaded CSV files and prepare for fine-tuning"""
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(raw_data_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files to process.")
    
    # Process each file
    all_texts = []
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"Processing {filename}...")
        
        try:
            # Read CSV file with more robust error handling
            try:
                # First try standard parsing
                df = pd.read_csv(csv_file, encoding='utf-8')
            except pd.errors.ParserError:
                # If that fails, try with quoting
                try:
                    df = pd.read_csv(csv_file, encoding='utf-8', quoting=1)  # QUOTE_ALL
                except pd.errors.ParserError:
                    # If that still fails, try with the Python engine which is more forgiving
                    try:
                        df = pd.read_csv(csv_file, encoding='utf-8', engine='python')
                    except Exception as e:
                        print(f"Failed to parse CSV with multiple methods: {e}")
                        # As a last resort, read the file line by line
                        with open(csv_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            if len(lines) > 0:
                                header = lines[0].strip()
                                if 'text' in header.lower():
                                    texts = [line.strip() for line in lines[1:] if line.strip()]
                                    print(f"Manually extracted {len(texts)} lines from {filename}.")
                                    all_texts.extend(texts)
                                    continue
                        # If we got here, we couldn't parse the file
                        continue
            
            # Check columns
            text_column = None
            for possible_column in ['text', 'content', 'Text', 'Content', 'TEXT', 'CONTENT']:
                if possible_column in df.columns:
                    text_column = possible_column
                    break
            
            if text_column is None:
                print(f"Warning: Could not find text column in {filename}. Printing column names:")
                print(df.columns.tolist())
                continue
            
            # Extract text data
            texts = df[text_column].dropna().tolist()
            print(f"Extracted {len(texts)} text samples from {filename}.")
            
            # Add to collection
            all_texts.extend(texts)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"Total extracted texts: {len(all_texts)}")
    
    # If no texts were extracted, create a minimal dataset
    if not all_texts:
        print("No texts were extracted from CSV files. Creating minimal sample dataset.")
        all_texts = [
            "שלום עולם! זוהי דוגמה לטקסט בעברית לצורך אימון מודל השפה.",
            "עברית היא שפה שמית, ממשפחת השפות האפרו-אסיאתיות, הנהוגה כשפה רשמית במדינת ישראל.",
            "תל אביב היא עיר במחוז תל אביב בישראל, ובירתו הכלכלית של המדינה.",
            "ירושלים היא בירת ישראל ועיר הבירה הגדולה ביותר במדינה.",
            "השפה העברית התפתחה לאורך אלפי שנים, מהעברית המקראית ועד לתחייתה בעת החדשה."
        ]
    
    # Save as JSONL for fine-tuning
    output_file = os.path.join(processed_dir, "hebrew_dataset.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in tqdm(all_texts, desc="Saving texts"):
            if text and len(text.strip()) > 50:  # Reduced minimum length to ensure we have data
                f.write(json.dumps({"text": text.strip()}, ensure_ascii=False) + '\n')
    
    print(f"Processed data saved to {output_file}")
    return output_file

def prepare_for_training(jsonl_file, output_dir):
    """Prepare the processed data for training using the prepare_dataset.py script"""
    dataset_dir = os.path.join(output_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create a command to run the prepare_dataset.py script
    cmd = f"python qwen_model/prepare_dataset.py --input_file {jsonl_file} --output_dir {dataset_dir} --format text"
    print(f"Running command: {cmd}")
    
    # Run the command
    os.system(cmd)
    
    print(f"Dataset prepared for training and saved to {dataset_dir}")
    print("\nTo use this dataset for fine-tuning, run:")
    print(f"python qwen_model/train.py --dataset_path {dataset_dir}/dataset --config qwen_model/finetuning/training_config.json")

def main():
    args = parse_args()
    
    print("\n=== Downloading and Processing S3 Data for Fine-tuning ===\n")
    
    # Configure AWS and get session
    session = configure_aws(args)
    
    # Download files from S3
    raw_data_dir = download_s3_files(args, session)
    
    # Process CSV files
    jsonl_file = process_csv_files(raw_data_dir, args.output_dir)
    
    # Prepare for training
    prepare_for_training(jsonl_file, args.output_dir)

if __name__ == "__main__":
    main()