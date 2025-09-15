#!/usr/bin/env python3
"""
Fixed Hebrew Benchmark Runner Script
Runs Hebrew benchmarks based on configuration file and generates CSV results
"""

import json
import subprocess
import os
import sys
import pandas as pd
from pathlib import Path
import time
import logging
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HebrewBenchmarkRunner:
    def __init__(self, config_path: str = "config-benchmarks.json", base_dir: str = "."):
        self.config_path = config_path
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "benchmark_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Model path mappings
        self.model_paths = {
            "qwen3_32b": "Qwen/Qwen2.5-32B-Instruct",  # Hugging Face model
            "model260000": "/home/ec2-user/qwen-hebrew-finetuning/model260000"
        }
        
        # CORRECT Data path for Hebrew benchmarks
        self.data_base_path = "/home/ec2-user/qwen-hebrew-finetuning/heb_bnch/bnch_data"
        
        # Load configuration
        self.config = self.load_config()
        
        # Results storage
        self.results = {}
        
    def load_config(self) -> Dict:
        """Load benchmark configuration from JSON file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found!")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration JSON: {e}")
            sys.exit(1)
    
    def get_enabled_models(self) -> List[str]:
        """Get list of enabled models from configuration"""
        enabled = [model for model, enabled in self.config["models"].items() if enabled]
        logger.info(f"Enabled models: {enabled}")
        return enabled
    
    def get_enabled_benchmarks(self) -> List[str]:
        """Get list of enabled benchmarks from configuration"""
        enabled = [benchmark for benchmark, enabled in self.config["benchmarks"].items() if enabled]
        logger.info(f"Enabled benchmarks: {enabled}")
        return enabled
    
    def verify_data_files(self) -> bool:
        """Verify that all required data files exist"""
        enabled_benchmarks = self.get_enabled_benchmarks()
        missing_files = []
        
        logger.info(f"Checking data files in: {self.data_base_path}")
        
        for benchmark in enabled_benchmarks:
            benchmark_dir = Path(self.data_base_path) / benchmark
            train_file = benchmark_dir / "train.jsonl"
            validation_file = benchmark_dir / "validation.jsonl"
            
            logger.info(f"Checking {benchmark}:")
            logger.info(f"  Looking for: {train_file}")
            logger.info(f"  Looking for: {validation_file}")
            
            if not train_file.exists():
                missing_files.append(str(train_file))
            if not validation_file.exists():
                missing_files.append(str(validation_file))
        
        if missing_files:
            logger.error("Missing data files:")
            for file in missing_files:
                logger.error(f"  - {file}")
            return False
        
        logger.info("All required data files found")
        return True
    
    def setup_custom_tasks_import(self):
        """Setup custom tasks for import by modifying sys.path and environment"""
        heb_bnch_path = "/home/ec2-user/qwen-hebrew-finetuning/heb_bnch"
        if heb_bnch_path not in sys.path:
            sys.path.insert(0, heb_bnch_path)
        
        # Set environment variable for data path - CORRECT PATH
        os.environ["HEB_BNCH_DATA_PATH"] = self.data_base_path
        
        # Try to import custom tasks to verify they work
        try:
            import custom_tasks
            logger.info("Successfully imported custom_tasks module")
            
            # Also verify that the custom_tasks module can find the data
            if hasattr(custom_tasks, 'verify_data_paths'):
                if not custom_tasks.verify_data_paths():
                    logger.error("Custom tasks module cannot find data files")
                    return False
            
            return True
        except ImportError as e:
            logger.error(f"Failed to import custom_tasks: {e}")
            return False
    
    def build_lighteval_command(self, model_name: str, benchmark: str, model_path: str) -> List[str]:
        """Build the lighteval command for a specific model and benchmark"""
        
        # Create unique output directory for this combination
        output_dir = self.results_dir / f"{model_name}_{benchmark}_results"
        results_file = self.results_dir / f"{model_name}_{benchmark}_results.json"
        
        # Base command with proper custom tasks handling
        cmd = [
            "python", "-m", "lighteval", "accelerate",
            f"model_name={model_path},device=cuda:0,batch_size=1,dtype=bfloat16,generation_parameters={{top_k:1,temperature:1.0}}",
            f"community|{benchmark}|5|0",
            "--custom-tasks", "custom_tasks",
            "--output-dir", str(output_dir),
            "--save-details",
            "--results-path-template", str(results_file),
            "--max-samples", "25"
        ]
        
        return cmd
    
    def run_single_benchmark(self, model_name: str, benchmark: str) -> Tuple[bool, float]:
        """Run a single benchmark and return success status and accuracy"""
        
        if model_name not in self.model_paths:
            logger.error(f"Unknown model: {model_name}")
            return False, 0.0
            
        model_path = self.model_paths[model_name]
        
        # Check if model path exists (only for local models, not HF models)
        if not model_path.startswith(("Qwen/", "microsoft/", "google/", "meta-llama/", "mistralai/")) and not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            return False, 0.0
        
        # Verify benchmark data files exist
        benchmark_dir = Path(self.data_base_path) / benchmark
        if not benchmark_dir.exists():
            logger.error(f"Benchmark data directory does not exist: {benchmark_dir}")
            return False, 0.0
        
        logger.info(f"Running {benchmark} on {model_name}")
        
        # Build command
        cmd = self.build_lighteval_command(model_name, benchmark, model_path)
        
        # Set environment with proper Python path and data path - CORRECT PATH
        env = os.environ.copy()
        env["PYTHONPATH"] = "/home/ec2-user/qwen-hebrew-finetuning/heb_bnch"
        env["HEB_BNCH_DATA_PATH"] = self.data_base_path  # This is now the correct path
        
        # Also set HF_DATASETS_CACHE to help with data loading
        env["HF_DATASETS_CACHE"] = "/home/ec2-user/.cache/huggingface/datasets"
        
        try:
            # Run the command
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, 
                cwd="/home/ec2-user/qwen-hebrew-finetuning/heb_bnch",
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully completed {benchmark} on {model_name}")
                
                # Try to extract accuracy from results
                accuracy = self.extract_accuracy(model_name, benchmark)
                return True, accuracy
                
            else:
                logger.error(f"Failed to run {benchmark} on {model_name}")
                logger.error(f"Return code: {result.returncode}")
                if result.stdout:
                    logger.error(f"stdout: {result.stdout[-2000:]}")  # Last 2000 chars
                if result.stderr:
                    logger.error(f"stderr: {result.stderr[-2000:]}")  # Last 2000 chars
                return False, 0.0
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout running {benchmark} on {model_name}")
            return False, 0.0
        except Exception as e:
            logger.error(f"Error running {benchmark} on {model_name}: {e}")
            return False, 0.0
    
    def extract_accuracy(self, model_name: str, benchmark: str) -> float:
        """Extract accuracy from results file"""
        results_file = self.results_dir / f"{model_name}_{benchmark}_results.json"
        
        try:
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
                
                # Navigate through the results structure to find accuracy
                if "results" in results_data:
                    for task_name, task_results in results_data["results"].items():
                        if benchmark in task_name:
                            # Look for different accuracy metric names
                            for metric_name, metric_value in task_results.items():
                                if "acc" in metric_name.lower() or "accuracy" in metric_name.lower():
                                    logger.info(f"Found accuracy for {model_name}/{benchmark}: {metric_value}")
                                    return float(metric_value)
                            
                            # If no accuracy found, try to get the first numeric metric
                            for metric_name, metric_value in task_results.items():
                                try:
                                    value = float(metric_value)
                                    if 0 <= value <= 1:  # Assuming accuracy is between 0 and 1
                                        logger.info(f"Using metric {metric_name} as accuracy for {model_name}/{benchmark}: {value}")
                                        return value
                                except (ValueError, TypeError):
                                    continue
                
                logger.warning(f"Could not extract accuracy from {results_file}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error reading results file {results_file}: {e}")
            return 0.0
        
        return 0.0
    
    def run_all_benchmarks(self):
        """Run all enabled benchmarks on all enabled models"""
        # First, verify data files exist
        if not self.verify_data_files():
            logger.error("Cannot proceed without required data files. Please check your data directory.")
            return
        
        # Setup custom tasks import
        if not self.setup_custom_tasks_import():
            logger.error("Cannot proceed without custom tasks. Please check your custom_tasks module.")
            return
        
        enabled_models = self.get_enabled_models()
        enabled_benchmarks = self.get_enabled_benchmarks()
        
        if not enabled_models:
            logger.error("No models enabled in configuration")
            return
            
        if not enabled_benchmarks:
            logger.error("No benchmarks enabled in configuration")
            return
        
        total_runs = len(enabled_models) * len(enabled_benchmarks)
        current_run = 0
        
        logger.info(f"Starting benchmark runs: {len(enabled_models)} models × {len(enabled_benchmarks)} benchmarks = {total_runs} total runs")
        
        for model_name in enabled_models:
            self.results[model_name] = {}
            
            for benchmark in enabled_benchmarks:
                current_run += 1
                logger.info(f"Progress: {current_run}/{total_runs} - Running {benchmark} on {model_name}")
                
                success, accuracy = self.run_single_benchmark(model_name, benchmark)
                
                if success:
                    self.results[model_name][benchmark] = accuracy
                    logger.info(f"✓ {model_name}/{benchmark}: {accuracy:.4f}")
                else:
                    self.results[model_name][benchmark] = None
                    logger.warning(f"✗ {model_name}/{benchmark}: FAILED")
                
                # Add small delay between runs
                time.sleep(2)
        
        logger.info("All benchmark runs completed!")
    
    def generate_csv_results(self, output_file: str = "benchmark_results.csv"):
        """Generate CSV file with results"""
        if not self.results:
            logger.error("No results to save")
            return
        
        # Create DataFrame
        enabled_benchmarks = self.get_enabled_benchmarks()
        
        df_data = []
        for model_name, model_results in self.results.items():
            row = {"model": model_name}
            for benchmark in enabled_benchmarks:
                accuracy = model_results.get(benchmark)
                if accuracy is not None:
                    row[benchmark] = f"{accuracy:.4f}"
                else:
                    row[benchmark] = "FAILED"
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save CSV
        output_path = self.results_dir / output_file
        df.to_csv(output_path, index=False)
        
        logger.info(f"Results saved to: {output_path}")
        
        # Display results
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        return output_path
    
    def save_detailed_results(self, output_file: str = "detailed_results.json"):
        """Save detailed results as JSON"""
        output_path = self.results_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed results saved to: {output_path}")
        return output_path
    
    def create_debug_script(self):
        """Create a debug script to check data paths and custom tasks"""
        debug_script = f"""#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add to Python path
sys.path.insert(0, "/home/ec2-user/qwen-hebrew-finetuning/heb_bnch")

# Set environment variables - CORRECT PATH
os.environ["HEB_BNCH_DATA_PATH"] = "/home/ec2-user/qwen-hebrew-finetuning/heb_bnch/bnch_data"
os.environ["PYTHONPATH"] = "/home/ec2-user/qwen-hebrew-finetuning/heb_bnch"

print("Checking data files...")
data_path = Path("/home/ec2-user/qwen-hebrew-finetuning/heb_bnch/bnch_data")
benchmarks = ["arc_ai2_heb", "mmlu_heb", "copa_heb", "hellaswag_heb", "psychometric_heb_math"]

for benchmark in benchmarks:
    # Handle different folder structures
    if benchmark.startswith("psychometric_heb_"):
        subject = benchmark.replace("psychometric_heb_", "")
        benchmark_dir = data_path / "psychometric_heb" / subject
    else:
        benchmark_dir = data_path / benchmark
        
    train_file = benchmark_dir / "train.jsonl"
    validation_file = benchmark_dir / "validation.jsonl"
    
    print(f"\\n{{benchmark}}:")
    print(f"  Directory exists: {{benchmark_dir.exists()}}")
    print(f"  train.jsonl exists: {{train_file.exists()}}")
    print(f"  validation.jsonl exists: {{validation_file.exists()}}")
    
    if benchmark_dir.exists():
        files = list(benchmark_dir.glob("*"))
        print(f"  Files in directory: {{[f.name for f in files]}}")

print("\\nTrying to import custom_tasks...")
try:
    import custom_tasks
    print("✓ Successfully imported custom_tasks")
    
    # Try to access task definitions
    if hasattr(custom_tasks, '__dict__'):
        tasks = [attr for attr in dir(custom_tasks) if not attr.startswith('_')]
        print(f"Available tasks: {{tasks}}")
        
except ImportError as e:
    print(f"✗ Failed to import custom_tasks: {{e}}")

print("\\nDone!")
"""
        
        debug_path = self.results_dir / "debug_data_paths.py"
        with open(debug_path, 'w') as f:
            f.write(debug_script)
        
        os.chmod(debug_path, 0o755)
        logger.info(f"Debug script created: {debug_path}")
        logger.info(f"Run it with: python {debug_path}")
        
        return debug_path


def main():
    """Main function"""
    # Change to the correct working directory
    os.chdir("/home/ec2-user/qwen-hebrew-finetuning/heb_bnch")
    
    print("Hebrew Benchmark Runner")
    print("="*50)
    
    # Initialize runner
    runner = HebrewBenchmarkRunner()
    
    # Create debug script first
    debug_script = runner.create_debug_script()
    print(f"\nDebug script created at: {debug_script}")
    print("Run this first to check your setup: python", debug_script)
    
    response = input("\nDo you want to run the debug script first? (y/n): ")
    if response.lower() == 'y':
        try:
            subprocess.run([sys.executable, str(debug_script)], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Debug script failed: {e}")
            return
    
    # Run all benchmarks
    runner.run_all_benchmarks()
    
    # Generate results
    csv_path = runner.generate_csv_results()
    json_path = runner.save_detailed_results()
    
    print(f"\nResults saved:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()