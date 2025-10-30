#!/usr/bin/env python3
"""
S3 Benchmark Dataset Integration Script
Downloads psychometric benchmark dataset from S3 and integrates it with the project
"""

import os
import boto3
import pandas as pd
import argparse
from pathlib import Path
import json
from typing import Dict, List, Any

class S3BenchmarkConnector:
    def __init__(self, aws_access_key_id: str, aws_secret_access_key: str, region_name: str = "us-east-1"):
        """
        Initialize S3 connector for benchmark dataset
        
        Args:
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            region_name: AWS region name
        """
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name
        )
        
        # Create benchmarks directory
        self.benchmarks_dir = Path("benchmarks")
        self.benchmarks_dir.mkdir(exist_ok=True)
        
        print(f"✅ S3 client initialized for region: {self.region_name}")

    def download_psychometric_dataset(self, bucket_name: str = "gepeta-datasets", 
                                    s3_key: str = "benchmarks/psychometric/psychometric_dataset.xlsx"):
        """
        Download psychometric dataset from S3
        
        Args:
            bucket_name: S3 bucket name
            s3_key: S3 object key
            
        Returns:
            Path to downloaded file
        """
        try:
            # Create local file path
            local_file_path = self.benchmarks_dir / "psychometric_dataset.xlsx"
            
            print(f"📥 Downloading {s3_key} from {bucket_name}...")
            
            # Download file from S3
            self.s3_client.download_file(
                Bucket=bucket_name,
                Key=s3_key,
                Filename=str(local_file_path)
            )
            
            print(f"✅ Successfully downloaded to: {local_file_path}")
            print(f"📊 File size: {local_file_path.stat().st_size / (1024*1024):.2f} MB")
            
            return local_file_path
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return None

    def process_psychometric_data(self, file_path: Path) -> Dict[str, Any]:
        """
        Process the psychometric Excel dataset
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary with processed data and metadata
        """
        try:
            print(f"📊 Processing psychometric dataset: {file_path}")
            
            # Read Excel file
            df = pd.read_excel(file_path)
            
            # Basic dataset info
            dataset_info = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": df.columns.tolist(),
                "sample_data": df.head(3).to_dict('records') if len(df) > 0 else [],
                "data_types": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict()
            }
            
            print(f"✅ Dataset processed successfully:")
            print(f"   📈 Rows: {dataset_info['total_rows']}")
            print(f"   📋 Columns: {dataset_info['total_columns']}")
            print(f"   🔤 Column names: {', '.join(dataset_info['columns'][:5])}...")
            
            # Save processed data as JSON
            json_file_path = self.benchmarks_dir / "psychometric_dataset_info.json"
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"💾 Dataset info saved to: {json_file_path}")
            
            return dataset_info
            
        except Exception as e:
            print(f"❌ Error processing dataset: {e}")
            return {}

    def convert_to_evaluation_format(self, file_path: Path) -> Path:
        """
        Convert psychometric dataset to LightEval compatible format
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Path to converted JSONL file
        """
        try:
            print(f"🔄 Converting dataset to evaluation format...")
            
            # Read Excel file
            df = pd.read_excel(file_path)
            
            # Convert to LightEval format
            eval_data = []
            
            for idx, row in df.iterrows():
                # Assuming the dataset has question, options, and answer columns
                # Adjust column names based on actual dataset structure
                eval_item = {
                    "id": f"psychometric_{idx}",
                    "question": str(row.get('question', row.get('Question', ''))),
                    "choices": self._extract_choices(row),
                    "answer": str(row.get('answer', row.get('Answer', ''))),
                    "subject": "psychometric",
                    "language": "hebrew"
                }
                eval_data.append(eval_item)
            
            # Save as JSONL for LightEval
            jsonl_file_path = self.benchmarks_dir / "psychometric_dataset.jsonl"
            with open(jsonl_file_path, 'w', encoding='utf-8') as f:
                for item in eval_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"✅ Converted dataset saved to: {jsonl_file_path}")
            print(f"📊 Total evaluation items: {len(eval_data)}")
            
            return jsonl_file_path
            
        except Exception as e:
            print(f"❌ Error converting dataset: {e}")
            return None

    def _extract_choices(self, row: pd.Series) -> List[str]:
        """
        Extract multiple choice options from row
        
        Args:
            row: Pandas Series representing a row
            
        Returns:
            List of choice options
        """
        choices = []
        
        # Common column patterns for multiple choice
        choice_patterns = ['A', 'B', 'C', 'D', 'E', 'option_a', 'option_b', 'option_c', 'option_d']
        
        for pattern in choice_patterns:
            if pattern in row and pd.notna(row[pattern]):
                choices.append(str(row[pattern]))
        
        # If no standard patterns found, try to find any columns with choices
        if not choices:
            for col in row.index:
                if 'choice' in col.lower() or 'option' in col.lower():
                    if pd.notna(row[col]):
                        choices.append(str(row[col]))
        
        return choices

    def integrate_with_evaluation(self, jsonl_file_path: Path):
        """
        Integrate the dataset with the existing evaluation pipeline
        
        Args:
            jsonl_file_path: Path to the JSONL evaluation file
        """
        try:
            print(f"🔗 Integrating with evaluation pipeline...")
            
            # Create evaluation configuration
            eval_config = {
                "dataset_name": "psychometric_hebrew",
                "dataset_path": str(jsonl_file_path),
                "language": "hebrew",
                "task_type": "multiple_choice",
                "metrics": ["accuracy", "f1"],
                "description": "Hebrew psychometric reasoning benchmark"
            }
            
            # Save evaluation config
            config_file_path = self.benchmarks_dir / "psychometric_eval_config.json"
            with open(config_file_path, 'w', encoding='utf-8') as f:
                json.dump(eval_config, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Evaluation config saved to: {config_file_path}")
            
            # Create evaluation script
            self._create_evaluation_script()
            
        except Exception as e:
            print(f"❌ Error integrating with evaluation: {e}")

    def _create_evaluation_script(self):
        """Create a script to run psychometric evaluation"""
        
        eval_script_content = '''#!/usr/bin/env python3
"""
Psychometric Benchmark Evaluation Script
"""

import json
import sys
from pathlib import Path

def run_psychometric_evaluation(model_path: str = None):
    """
    Run psychometric benchmark evaluation
    
    Args:
        model_path: Path to the fine-tuned model
    """
    try:
        # Load evaluation config
        config_path = Path("benchmarks/psychometric_eval_config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"🚀 Running psychometric evaluation...")
        print(f"📊 Dataset: {config['dataset_name']}")
        print(f"📁 Dataset path: {config['dataset_path']}")
        
        # TODO: Integrate with LightEval
        # This is a placeholder for the actual evaluation logic
        print("⚠️  Evaluation integration pending - dataset is ready for LightEval")
        
        return True
        
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return False

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "qwen_model/finetuned"
    run_psychometric_evaluation(model_path)
'''
        
        script_path = self.benchmarks_dir / "run_psychometric_eval.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(eval_script_content)
        
        # Make script executable
        script_path.chmod(0o755)
        
        print(f"✅ Evaluation script created: {script_path}")

def main():
    parser = argparse.ArgumentParser(description="Connect S3 benchmark dataset")
    parser.add_argument("--aws_access_key_id", required=True, help="AWS Access Key ID")
    parser.add_argument("--aws_secret_access_key", required=True, help="AWS Secret Access Key")
    parser.add_argument("--region", default="us-east-1", help="AWS Region")
    parser.add_argument("--bucket", default="gepeta-datasets", help="S3 Bucket name")
    parser.add_argument("--s3_key", default="benchmarks/psychometric/psychometric_dataset.xlsx", help="S3 object key")
    
    args = parser.parse_args()
    
    print("🚀 Starting S3 Benchmark Dataset Connection...")
    
    # Initialize connector
    connector = S3BenchmarkConnector(
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        region_name=args.region
    )
    
    # Download dataset
    file_path = connector.download_psychometric_dataset(
        bucket_name=args.bucket,
        s3_key=args.s3_key
    )
    
    if file_path:
        # Process dataset
        dataset_info = connector.process_psychometric_data(file_path)
        
        if dataset_info:
            # Convert to evaluation format
            jsonl_path = connector.convert_to_evaluation_format(file_path)
            
            if jsonl_path:
                # Integrate with evaluation pipeline
                connector.integrate_with_evaluation(jsonl_path)
                
                print("\n🎉 Benchmark dataset successfully connected!")
                print(f"📁 Files created:")
                print(f"   📊 Original dataset: {file_path}")
                print(f"   📋 Dataset info: benchmarks/psychometric_dataset_info.json")
                print(f"   🔄 Evaluation format: {jsonl_path}")
                print(f"   ⚙️  Evaluation config: benchmarks/psychometric_eval_config.json")
                print(f"   🚀 Evaluation script: benchmarks/run_psychometric_eval.py")
                
                print(f"\n📝 Next steps:")
                print(f"   1. Review dataset info: cat benchmarks/psychometric_dataset_info.json")
                print(f"   2. Run evaluation: python benchmarks/run_psychometric_eval.py")
                print(f"   3. Integrate with LightEval for model evaluation")

if __name__ == "__main__":
    main()
