#!/usr/bin/env python3
"""
Psychometric Benchmark Integration with LightEval
Integrates the Hebrew psychometric dataset with the existing evaluation pipeline
"""

import json
import pandas as pd
from pathlib import Path
import argparse
import sys
from typing import Dict, List, Any

class PsychometricBenchmarkIntegrator:
    def __init__(self):
        """Initialize the benchmark integrator"""
        self.benchmarks_dir = Path("benchmarks")
        self.dataset_path = self.benchmarks_dir / "psychometric_dataset.xlsx"
        self.jsonl_path = self.benchmarks_dir / "psychometric_dataset.jsonl"
        
    def create_lighteval_task(self):
        """Create LightEval task configuration for psychometric benchmark"""
        
        task_config = {
            "group": "psychometric_hebrew",
            "dataset": {
                "path": str(self.jsonl_path),
                "name": "psychometric_hebrew"
            },
            "task": "psychometric_hebrew",
            "metric": ["accuracy", "exact_match"],
            "output_type": "multiple_choice",
            "few_shot_split": "train",
            "few_shot_select": "sequential",
            "few_shot": 5,
            "generation_size": 1,
            "stop_sequence": ["\n"],
            "trust_dataset": True,
            "version": 1.0
        }
        
        # Save task configuration
        task_config_path = self.benchmarks_dir / "lighteval_psychometric_task.json"
        with open(task_config_path, 'w', encoding='utf-8') as f:
            json.dump(task_config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ LightEval task config created: {task_config_path}")
        return task_config_path
    
    def improve_jsonl_format(self):
        """Improve the JSONL format to be more compatible with LightEval"""
        
        # Read the original Excel file for better processing
        df = pd.read_excel(self.dataset_path)
        
        improved_data = []
        
        for idx, row in df.iterrows():
            # Extract question and context
            question = str(row.get('Question', '')).strip()
            context = str(row.get('Context', '')).strip() if pd.notna(row.get('Context')) else ""
            
            # Extract options
            options = []
            for i in range(1, 5):  # Options 1-4
                option = row.get(f'Option {i}', '')
                if pd.notna(option) and str(option).strip():
                    options.append(str(option).strip())
            
            # Get correct answer
            answer = str(row.get('Answer', '')).strip()
            
            # Create LightEval compatible format
            if question and options and answer:
                eval_item = {
                    "input": f"{context}\n\n{question}" if context else question,
                    "target": answer,
                    "choices": options,
                    "gold": self._get_answer_index(answer, options),
                    "subject": "psychometric",
                    "level": "undergraduate",
                    "language": "hebrew",
                    "question_type": str(row.get('Question Type', '')),
                    "chapter": str(row.get('Chapter Name', '')),
                    "id": f"psychometric_{idx}"
                }
                improved_data.append(eval_item)
        
        # Save improved JSONL
        improved_jsonl_path = self.benchmarks_dir / "psychometric_lighteval.jsonl"
        with open(improved_jsonl_path, 'w', encoding='utf-8') as f:
            for item in improved_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ Improved JSONL format created: {improved_jsonl_path}")
        print(f"📊 Total items: {len(improved_data)}")
        
        return improved_jsonl_path, len(improved_data)
    
    def _get_answer_index(self, answer: str, options: List[str]) -> int:
        """Get the index of the correct answer in the options list"""
        answer = answer.strip()
        
        # Try to match by number (1, 2, 3, 4)
        try:
            answer_num = int(answer)
            if 1 <= answer_num <= len(options):
                return answer_num - 1
        except ValueError:
            pass
        
        # Try to match by letter (A, B, C, D)
        if answer.upper() in ['A', 'B', 'C', 'D']:
            letter_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            return letter_to_index.get(answer.upper(), 0)
        
        # Try to match by exact text
        for i, option in enumerate(options):
            if answer.lower() in option.lower() or option.lower() in answer.lower():
                return i
        
        return 0  # Default to first option if no match found
    
    def create_evaluation_script(self):
        """Create a comprehensive evaluation script"""
        
        eval_script = '''#!/usr/bin/env python3
"""
Run Psychometric Benchmark Evaluation with LightEval
"""

import os
import subprocess
import sys
from pathlib import Path

def run_psychometric_evaluation(model_path: str, output_dir: str = "eval_results"):
    """
    Run psychometric benchmark evaluation using LightEval
    
    Args:
        model_path: Path to the model to evaluate
        output_dir: Directory to save evaluation results
    """
    
    print("🚀 Starting Psychometric Benchmark Evaluation...")
    
    # Paths
    benchmark_dir = Path("benchmarks")
    jsonl_path = benchmark_dir / "psychometric_lighteval.jsonl"
    task_config_path = benchmark_dir / "lighteval_psychometric_task.json"
    
    if not jsonl_path.exists():
        print(f"❌ Dataset not found: {jsonl_path}")
        print("Please run: python integrate_psychometric_benchmark.py --improve-format")
        return False
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # LightEval command
    cmd = [
        "lighteval",
        "--model_args", f"pretrained={model_path}",
        "--tasks", "psychometric_hebrew",
        "--output_dir", str(output_path),
        "--custom_task_file", str(task_config_path),
        "--dataset_path", str(jsonl_path),
        "--batch_size", "4",
        "--num_fewshot", "5"
    ]
    
    print(f"📊 Running evaluation command:")
    print(f"   {' '.join(cmd)}")
    
    try:
        # Run evaluation
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: {output_path}")
        
        if result.stdout:
            print("📊 Evaluation output:")
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ LightEval not found. Please install it first:")
        print("   uv pip install lighteval")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run psychometric benchmark evaluation")
    parser.add_argument("--model_path", default="qwen_model/finetuned", 
                       help="Path to the model to evaluate")
    parser.add_argument("--output_dir", default="eval_results/psychometric", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    success = run_psychometric_evaluation(args.model_path, args.output_dir)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
'''
        
        script_path = self.benchmarks_dir / "run_psychometric_lighteval.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(eval_script)
        
        script_path.chmod(0o755)
        print(f"✅ LightEval evaluation script created: {script_path}")
        
        return script_path
    
    def update_makefile(self):
        """Update the Makefile to include psychometric evaluation"""
        
        makefile_path = Path("Makefile")
        
        # Additional Makefile targets for psychometric evaluation
        psychometric_targets = '''
# Psychometric benchmark evaluation
eval-psychometric:
\t@echo "🧠 Running psychometric benchmark evaluation..."
\t@source .venv/bin/activate && python benchmarks/run_psychometric_lighteval.py

setup-psychometric:
\t@echo "🔧 Setting up psychometric benchmark..."
\t@source .venv/bin/activate && python integrate_psychometric_benchmark.py --improve-format

# Download benchmark from S3 (requires AWS credentials)
download-psychometric:
\t@echo "📥 Downloading psychometric benchmark from S3..."
\t@source .venv/bin/activate && python connect_s3_benchmark.py \\
\t\t--aws_access_key_id $(AWS_ACCESS_KEY_ID) \\
\t\t--aws_secret_access_key $(AWS_SECRET_ACCESS_KEY)
'''
        
        # Add to existing Makefile
        if makefile_path.exists():
            with open(makefile_path, 'a') as f:
                f.write(psychometric_targets)
            print(f"✅ Updated Makefile with psychometric targets")
        else:
            print(f"⚠️  Makefile not found, creating psychometric-only Makefile")
            with open(makefile_path, 'w') as f:
                f.write(psychometric_targets)

def main():
    parser = argparse.ArgumentParser(description="Integrate psychometric benchmark with LightEval")
    parser.add_argument("--improve-format", action="store_true", 
                       help="Improve JSONL format for better LightEval compatibility")
    parser.add_argument("--create-task", action="store_true", 
                       help="Create LightEval task configuration")
    parser.add_argument("--create-scripts", action="store_true", 
                       help="Create evaluation scripts")
    parser.add_argument("--update-makefile", action="store_true", 
                       help="Update Makefile with psychometric targets")
    parser.add_argument("--all", action="store_true", 
                       help="Run all integration steps")
    
    args = parser.parse_args()
    
    integrator = PsychometricBenchmarkIntegrator()
    
    if args.all:
        args.improve_format = True
        args.create_task = True
        args.create_scripts = True
        args.update_makefile = True
    
    if args.improve_format:
        print("🔄 Improving JSONL format...")
        improved_path, count = integrator.improve_jsonl_format()
        
    if args.create_task:
        print("⚙️ Creating LightEval task configuration...")
        task_config_path = integrator.create_lighteval_task()
        
    if args.create_scripts:
        print("📝 Creating evaluation scripts...")
        script_path = integrator.create_evaluation_script()
        
    if args.update_makefile:
        print("📋 Updating Makefile...")
        integrator.update_makefile()
    
    if args.all:
        print("\n🎉 Psychometric benchmark integration complete!")
        print("\n📝 Usage:")
        print("   make setup-psychometric    # Setup benchmark format")
        print("   make eval-psychometric     # Run evaluation")
        print("   python benchmarks/run_psychometric_lighteval.py --model_path your_model")

if __name__ == "__main__":
    main()
