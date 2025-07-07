#!/usr/bin/env python3
"""
Automated benchmark runner for P4/P5 instance comparison
Submits training jobs across different instance types and collects performance metrics
"""

import boto3
import json
import time
import pandas as pd
from datetime import datetime, timedelta
import logging
import argparse
from typing import Dict, List, Any
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SageMakerBenchmarkRunner:
    """Automated benchmark runner for P4/P5 instance comparison"""
    
    def __init__(self, role_arn: str, bucket_name: str, region: str = 'us-east-1'):
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.role_arn = role_arn
        self.bucket_name = bucket_name
        self.region = region
        self.results = []
        
        # Get account ID for ECR image URI
        sts = boto3.client('sts')
        self.account_id = sts.get_caller_identity()['Account']
        
    def get_training_image_uri(self) -> str:
        """Get the ECR training image URI"""
        return f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/qwen-hebrew-training:latest"
    
    def run_benchmark(self, instance_types: List[str], dataset_path: str, epochs: int = 1, max_steps: int = None) -> List[Dict]:
        """Run training jobs on all specified instance types"""
        logger.info(f"Starting benchmark across {len(instance_types)} instance types")
        logger.info(f"Instance types: {instance_types}")
        
        job_names = []
        
        for instance_type in instance_types:
            job_name = self._generate_job_name(instance_type)
            logger.info(f"Submitting training job: {job_name} on {instance_type}")
            
            try:
                self._submit_training_job(job_name, instance_type, dataset_path, epochs, max_steps)
                job_names.append((job_name, instance_type))
                
                # Wait a bit between submissions to avoid throttling
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Failed to submit job for {instance_type}: {e}")
                continue
        
        if not job_names:
            logger.error("No training jobs were successfully submitted")
            return []
        
        logger.info(f"Successfully submitted {len(job_names)} training jobs")
        return self._monitor_jobs(job_names)
    
    def _generate_job_name(self, instance_type: str) -> str:
        """Generate unique job name"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        instance_clean = instance_type.replace('.', '-').replace('ml-', '')
        return f"qwen-benchmark-{instance_clean}-{timestamp}"
    
    def _submit_training_job(self, job_name: str, instance_type: str, dataset_path: str, epochs: int, max_steps: int = None):
        """Submit a training job for specific instance type"""
        
        # Prepare hyperparameters
        hyperparameters = {
            'epochs': str(epochs),
            'instance-type': instance_type,
            'benchmark-mode': 'true',
            'wandb-project': 'qwen-hebrew-sagemaker-benchmark',
            'wandb-run-name': job_name
        }
        
        if max_steps:
            hyperparameters['max-steps'] = str(max_steps)
        
        training_job_config = {
            'TrainingJobName': job_name,
            'AlgorithmSpecification': {
                'TrainingImage': self.get_training_image_uri(),
                'TrainingInputMode': 'File'
            },
            'RoleArn': self.role_arn,
            'Environment': {
                'WANDB_API_KEY': '70bbb9e8a69e33f73d1adfd7577ad0783b970dd6'  # W&B API Key
            },
            'InputDataConfig': [
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': dataset_path,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'application/jsonl',
                    'CompressionType': 'None'
                }
            ],
            'OutputDataConfig': {
                'S3OutputPath': f's3://{self.bucket_name}/models/{job_name}/'
            },
            'ResourceConfig': {
                'InstanceType': instance_type,
                'InstanceCount': 1,
                'VolumeSizeInGB': 500
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 86400  # 24 hours max
            },
            'HyperParameters': hyperparameters,
            'Tags': [
                {
                    'Key': 'Project',
                    'Value': 'QwenHebrewBenchmark'
                },
                {
                    'Key': 'InstanceType',
                    'Value': instance_type
                },
                {
                    'Key': 'BenchmarkRun',
                    'Value': datetime.now().strftime("%Y%m%d")
                }
            ]
        }
        
        logger.info('Calling self.sagemaker.create_training_job(**training_job_config)')
        response = self.sagemaker.create_training_job(**training_job_config)
        logger.info(f"Training job {job_name} submitted successfully")
        return response
    
    def _monitor_jobs(self, job_names: List[tuple]) -> List[Dict]:
        """Monitor training jobs and collect results"""
        logger.info(f"Monitoring {len(job_names)} training jobs...")
        
        completed_jobs = []
        start_time = time.time()
        
        while len(completed_jobs) < len(job_names):
            # Check timeout (24 hours)
            if time.time() - start_time > 86400:
                logger.warning("Monitoring timeout reached (24 hours)")
                break
            
            for job_name, instance_type in job_names:
                if job_name in [job['job_name'] for job in completed_jobs]:
                    continue
                
                try:
                    response = self.sagemaker.describe_training_job(TrainingJobName=job_name)
                    status = response['TrainingJobStatus']
                    
                    logger.info(f"Job {job_name} ({instance_type}): {status}")
                    
                    if status in ['Completed', 'Failed', 'Stopped']:
                        metrics = self._collect_job_metrics(job_name, instance_type, response)
                        completed_jobs.append(metrics)
                        logger.info(f"Collected metrics for {job_name}")
                
                except Exception as e:
                    logger.error(f"Error checking job {job_name}: {e}")
            
            # Wait before next check
            time.sleep(300)  # Check every 5 minutes
        
        logger.info(f"Monitoring completed. Collected metrics for {len(completed_jobs)} jobs")
        return completed_jobs
    
    def _collect_job_metrics(self, job_name: str, instance_type: str, job_description: Dict) -> Dict:
        """Collect comprehensive metrics from completed job"""
        
        # Basic job information
        metrics = {
            'job_name': job_name,
            'instance_type': instance_type,
            'status': job_description['TrainingJobStatus'],
            'creation_time': job_description['CreationTime'],
            'training_start_time': job_description.get('TrainingStartTime'),
            'training_end_time': job_description.get('TrainingEndTime'),
        }
        
        # Calculate training duration
        if metrics['training_start_time'] and metrics['training_end_time']:
            duration = metrics['training_end_time'] - metrics['training_start_time']
            metrics['training_duration_seconds'] = duration.total_seconds()
            metrics['training_duration_hours'] = duration.total_seconds() / 3600
        
        # Get instance configuration
        instance_config = self._get_instance_config(instance_type)
        metrics.update(instance_config)
        
        # Calculate cost
        if 'training_duration_hours' in metrics and 'estimated_hourly_cost' in metrics:
            metrics['actual_cost'] = metrics['training_duration_hours'] * metrics['estimated_hourly_cost']
        
        # Collect CloudWatch metrics
        cloudwatch_metrics = self._get_cloudwatch_metrics(job_name, metrics.get('training_start_time'), metrics.get('training_end_time'))
        metrics.update(cloudwatch_metrics)
        
        # Try to get training metrics from output
        training_metrics = self._get_training_output_metrics(job_name)
        metrics.update(training_metrics)
        
        # Calculate performance scores
        metrics.update(self._calculate_performance_scores(metrics))
        
        return metrics
    
    def _get_instance_config(self, instance_type: str) -> Dict:
        """Get instance configuration details"""
        config_map = {
            'ml.p4de.24xlarge': {
                'gpu_count': 8,
                'gpu_type': 'A100',
                'gpu_memory_gb': 80,
                'estimated_hourly_cost': 40.96
            },
            'ml.p5.48xlarge': {
                'gpu_count': 8,
                'gpu_type': 'H100',
                'gpu_memory_gb': 80,
                'estimated_hourly_cost': 98.32
            },
            'ml.p5e.48xlarge': {
                'gpu_count': 8,
                'gpu_type': 'H100',
                'gpu_memory_gb': 80,
                'estimated_hourly_cost': 98.32,
                'storage': 'NVMe SSD'
            },
            'ml.p5en.48xlarge': {
                'gpu_count': 8,
                'gpu_type': 'H100',
                'gpu_memory_gb': 80,
                'estimated_hourly_cost': 98.32,
                'networking': 'EFA Enhanced'
            }
        }
        
        return config_map.get(instance_type, {})
    
    def _get_cloudwatch_metrics(self, job_name: str, start_time, end_time) -> Dict:
        """Get CloudWatch metrics for the training job"""
        if not start_time or not end_time:
            return {}
        
        metrics = {}
        
        try:
            # Get GPU utilization
            response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/SageMaker',
                MetricName='GPUUtilization',
                Dimensions=[
                    {
                        'Name': 'TrainingJobName',
                        'Value': job_name
                    }
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,  # 5 minutes
                Statistics=['Average', 'Maximum']
            )
            
            if response['Datapoints']:
                gpu_utilization = [dp['Average'] for dp in response['Datapoints']]
                metrics['avg_gpu_utilization'] = sum(gpu_utilization) / len(gpu_utilization)
                metrics['max_gpu_utilization'] = max([dp['Maximum'] for dp in response['Datapoints']])
            
            # Get GPU memory utilization
            response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/SageMaker',
                MetricName='GPUMemoryUtilization',
                Dimensions=[
                    {
                        'Name': 'TrainingJobName',
                        'Value': job_name
                    }
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Average', 'Maximum']
            )
            
            if response['Datapoints']:
                gpu_memory = [dp['Average'] for dp in response['Datapoints']]
                metrics['avg_gpu_memory_utilization'] = sum(gpu_memory) / len(gpu_memory)
                metrics['max_gpu_memory_utilization'] = max([dp['Maximum'] for dp in response['Datapoints']])
        
        except Exception as e:
            logger.warning(f"Failed to get CloudWatch metrics for {job_name}: {e}")
        
        return metrics
    
    def _get_training_output_metrics(self, job_name: str) -> Dict:
        """Get training metrics from job output"""
        try:
            # Try to download training metrics file
            metrics_key = f"models/{job_name}/training_metrics.json"
            
            response = self.s3.get_object(Bucket=self.bucket_name, Key=metrics_key)
            metrics_data = json.loads(response['Body'].read().decode('utf-8'))
            
            return {
                'total_tokens': metrics_data.get('total_tokens', 0),
                'avg_tokens_per_second': metrics_data.get('avg_tokens_per_second', 0),
                'training_successful': metrics_data.get('training_successful', False)
            }
        
        except Exception as e:
            logger.warning(f"Failed to get training output metrics for {job_name}: {e}")
            return {}
    
    def _calculate_performance_scores(self, metrics: Dict) -> Dict:
        """Calculate performance and cost-effectiveness scores"""
        scores = {}
        
        # Performance score (tokens per second normalized)
        if 'avg_tokens_per_second' in metrics and metrics['avg_tokens_per_second'] > 0:
            # Normalize against expected P5 performance (8000 tokens/sec)
            scores['performance_score'] = min(metrics['avg_tokens_per_second'] / 8000, 1.0)
        
        # Cost effectiveness (performance per dollar)
        if 'performance_score' in scores and 'estimated_hourly_cost' in metrics:
            scores['cost_effectiveness'] = scores['performance_score'] / metrics['estimated_hourly_cost'] * 100
        
        # GPU efficiency score
        if 'avg_gpu_utilization' in metrics:
            scores['gpu_efficiency_score'] = metrics['avg_gpu_utilization'] / 100
        
        # Overall score (weighted combination)
        if all(key in scores for key in ['performance_score', 'cost_effectiveness', 'gpu_efficiency_score']):
            scores['overall_score'] = (
                scores['performance_score'] * 0.4 +
                scores['cost_effectiveness'] * 0.4 +
                scores['gpu_efficiency_score'] * 0.2
            )
        
        return scores
    
    def generate_comparison_report(self, results: List[Dict]) -> pd.DataFrame:
        """Generate comprehensive comparison report"""
        if not results:
            logger.warning("No results to generate report")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Add calculated columns
        if 'training_duration_hours' in df.columns and 'estimated_hourly_cost' in df.columns:
            df['cost_per_hour'] = df['estimated_hourly_cost']
            df['total_cost'] = df['training_duration_hours'] * df['estimated_hourly_cost']
        
        if 'avg_tokens_per_second' in df.columns and 'total_cost' in df.columns:
            df['tokens_per_dollar'] = df['avg_tokens_per_second'] / df['total_cost']
        
        # Sort by overall score
        if 'overall_score' in df.columns:
            df = df.sort_values('overall_score', ascending=False)
        
        return df
    
    def save_results(self, results: List[Dict], filename: str = None):
        """Save benchmark results to file and S3"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        # Save locally
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save to S3
        s3_key = f"benchmark_results/{filename}"
        self.s3.upload_file(filename, self.bucket_name, s3_key)
        
        logger.info(f"Results saved to {filename} and s3://{self.bucket_name}/{s3_key}")

def parse_args():
    parser = argparse.ArgumentParser(description="SageMaker P4/P5 Benchmark Runner")
    
    parser.add_argument('--role-arn', type=str, required=True, help='SageMaker execution role ARN')
    parser.add_argument('--bucket-name', type=str, required=True, help='S3 bucket for data and results')
    parser.add_argument('--dataset-path', type=str, required=True, help='S3 path to training dataset')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--max-steps', type=int, default=None, help='Maximum training steps for quick benchmark')
    parser.add_argument('--instance-types', type=str, nargs='+',
                       default=['ml.p4de.24xlarge', 'ml.p5.48xlarge', 'ml.p5e.48xlarge', 'ml.p5en.48xlarge'],
                       help='Instance types to benchmark')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    logger.info("Starting SageMaker P4/P5 benchmark comparison")
    logger.info(f"Instance types: {args.instance_types}")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Epochs: {args.epochs}")
    
    # Initialize benchmark runner
    runner = SageMakerBenchmarkRunner(
        role_arn=args.role_arn,
        bucket_name=args.bucket_name,
        region=args.region
    )
    
    # Run benchmark
    results = runner.run_benchmark(
        instance_types=args.instance_types,
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        max_steps=args.max_steps
    )
    
    if not results:
        logger.error("No benchmark results collected")
        return
    
    # Generate comparison report
    df = runner.generate_comparison_report(results)
    
    # Save results
    runner.save_results(results)
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    if not df.empty:
        # Display key metrics
        display_columns = [
            'instance_type', 'status', 'training_duration_hours', 
            'total_cost', 'avg_tokens_per_second', 'avg_gpu_utilization',
            'cost_effectiveness', 'overall_score'
        ]
        
        available_columns = [col for col in display_columns if col in df.columns]
        print(df[available_columns].to_string(index=False))
        
        # Recommendations
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        if 'overall_score' in df.columns:
            best_overall = df.iloc[0]
            print(f"Best Overall Performance: {best_overall['instance_type']}")
            print(f"  - Overall Score: {best_overall.get('overall_score', 'N/A'):.3f}")
            print(f"  - Cost: ${best_overall.get('total_cost', 'N/A'):.2f}")
            print(f"  - Performance: {best_overall.get('avg_tokens_per_second', 'N/A'):.0f} tokens/sec")
        
        if 'cost_effectiveness' in df.columns:
            best_cost = df.loc[df['cost_effectiveness'].idxmax()]
            print(f"\nMost Cost-Effective: {best_cost['instance_type']}")
            print(f"  - Cost Effectiveness: {best_cost['cost_effectiveness']:.3f}")
            print(f"  - Cost: ${best_cost.get('total_cost', 'N/A'):.2f}")
        
        if 'avg_tokens_per_second' in df.columns:
            fastest = df.loc[df['avg_tokens_per_second'].idxmax()]
            print(f"\nFastest Training: {fastest['instance_type']}")
            print(f"  - Speed: {fastest['avg_tokens_per_second']:.0f} tokens/sec")
            print(f"  - Duration: {fastest.get('training_duration_hours', 'N/A'):.2f} hours")
    
    print("\n" + "="*80)
    logger.info("Benchmark completed successfully!")

if __name__ == "__main__":
    main()