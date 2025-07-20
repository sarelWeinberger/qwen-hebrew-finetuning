#!/usr/bin/env python3
"""
SageMaker Job Management for Qwen Hebrew Fine-tuning
Provides high-level interface for submitting and managing training jobs
"""

import boto3
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class SageMakerJobManager:
    """High-level interface for SageMaker training jobs"""
    
    def __init__(self, role_arn: str, bucket_name: str, region: str = 'us-east-1'):
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.role_arn = role_arn
        self.bucket_name = bucket_name
        self.region = region
        
        # Get account ID for ECR
        sts = boto3.client('sts')
        self.account_id = sts.get_caller_identity()['Account']
    
    def submit_data_preparation_job(self, 
                                  job_name: str,
                                  s3_input_path: str,
                                  s3_output_path: str,
                                  instance_type: str = 'ml.m5.xlarge',
                                  **kwargs) -> str:
        """Submit data preparation job on CPU instance"""
        
        # Data preparation container
        image_uri = f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/qwen-data-prep:latest"
        
        job_config = {
            'TrainingJobName': job_name,
            'AlgorithmSpecification': {
                'TrainingImage': image_uri,
                'TrainingInputMode': 'File'
            },
            'RoleArn': self.role_arn,
            'InputDataConfig': [
                {
                    'ChannelName': 'raw-data',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': s3_input_path,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    }
                }
            ],
            'OutputDataConfig': {
                'S3OutputPath': s3_output_path
            },
            'ResourceConfig': {
                'InstanceType': instance_type,
                'InstanceCount': 1,
                'VolumeSizeInGB': 100
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 7200  # 2 hours
            },
            'HyperParameters': {
                'model-name': kwargs.get('model_name', 'Qwen/Qwen3-30B-A3B-Base'),
                'max-length': str(kwargs.get('max_length', 2048)),
                'train-split': str(kwargs.get('train_split', 0.9))
            },
            'Tags': [
                {'Key': 'JobType', 'Value': 'DataPreparation'},
                {'Key': 'Project', 'Value': 'QwenHebrew'}
            ]
        }
        
        response = self.sagemaker.create_training_job(**job_config)
        logger.info(f"Data preparation job {job_name} submitted")
        return job_name
    
    def submit_training_job(self,
                           job_name: str,
                           instance_type: str,
                           dataset_path: str,
                           epochs: int = 3,
                           max_steps: Optional[int] = None,
                           **kwargs) -> str:
        """Submit training job for specific instance type"""
        
        # Training container
        image_uri = f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/qwen-hebrew-training:latest"
        
        # Prepare hyperparameters
        hyperparameters = {
            'epochs': str(epochs),
            'instance-type': instance_type,
            'model-name': kwargs.get('model_name', 'Qwen/Qwen3-30B-A3B-Base'),
            'max-seq-length': str(kwargs.get('max_seq_length', 2048)),
            'seed': str(kwargs.get('seed', 42)),
            'wandb-project': kwargs.get('wandb_project', 'qwen-hebrew-sagemaker'),
            'wandb-run-name': kwargs.get('wandb_run_name', job_name)
        }
        
        if max_steps:
            hyperparameters['max-steps'] = str(max_steps)
        
        if kwargs.get('wandb_entity'):
            hyperparameters['wandb-entity'] = kwargs['wandb_entity']
        
        if kwargs.get('benchmark_mode'):
            hyperparameters['benchmark-mode'] = 'true'
        
        job_config = {
            'TrainingJobName': job_name,
            'AlgorithmSpecification': {
                'TrainingImage': image_uri,
                'TrainingInputMode': 'File'
            },
            'RoleArn': self.role_arn,
            'InputDataConfig': [
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': dataset_path,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    }
                }
            ],
            'OutputDataConfig': {
                'S3OutputPath': f's3://{self.bucket_name}/models/{job_name}/'
            },
            'ResourceConfig': {
                'InstanceType': instance_type,
                'InstanceCount': 1,
                'VolumeSizeInGB': kwargs.get('volume_size', 500)
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': kwargs.get('max_runtime', 86400)  # 24 hours
            },
            'HyperParameters': hyperparameters,
            'Tags': [
                {'Key': 'JobType', 'Value': 'Training'},
                {'Key': 'Project', 'Value': 'QwenHebrew'},
                {'Key': 'InstanceType', 'Value': instance_type}
            ]
        }
        
        # Add checkpoint configuration if specified
        if kwargs.get('checkpoint_s3_uri'):
            job_config['CheckpointConfig'] = {
                'S3Uri': kwargs['checkpoint_s3_uri'],
                'LocalPath': '/opt/ml/checkpoints'
            }
        
        response = self.sagemaker.create_training_job(**job_config)
        logger.info(f"Training job {job_name} submitted on {instance_type}")
        return job_name
    
    def submit_hyperparameter_tuning_job(self,
                                        tuning_job_name: str,
                                        instance_type: str,
                                        dataset_path: str,
                                        max_jobs: int = 10,
                                        max_parallel_jobs: int = 2,
                                        **kwargs) -> str:
        """Submit hyperparameter tuning job"""
        
        # Training container
        image_uri = f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/qwen-hebrew-training:latest"
        
        # Define hyperparameter ranges
        hyperparameter_ranges = {
            'learning_rate': {
                'Name': 'learning_rate',
                'Type': 'Continuous',
                'MinValue': '1e-6',
                'MaxValue': '1e-4',
                'ScalingType': 'Logarithmic'
            },
            'weight_decay': {
                'Name': 'weight_decay',
                'Type': 'Continuous',
                'MinValue': '0.01',
                'MaxValue': '0.1',
                'ScalingType': 'Linear'
            },
            'warmup_ratio': {
                'Name': 'warmup_ratio',
                'Type': 'Continuous',
                'MinValue': '0.0',
                'MaxValue': '0.2',
                'ScalingType': 'Linear'
            }
        }
        
        # Static hyperparameters
        static_hyperparameters = {
            'epochs': str(kwargs.get('epochs', 1)),
            'instance-type': instance_type,
            'model-name': kwargs.get('model_name', 'Qwen/Qwen3-30B-A3B-Base'),
            'max-seq-length': str(kwargs.get('max_seq_length', 2048)),
            'seed': str(kwargs.get('seed', 42)),
            'wandb-project': kwargs.get('wandb_project', 'qwen-hebrew-hp-tuning')
        }
        
        if kwargs.get('max_steps'):
            static_hyperparameters['max-steps'] = str(kwargs['max_steps'])
        
        tuning_config = {
            'HyperParameterTuningJobName': tuning_job_name,
            'HyperParameterTuningJobConfig': {
                'Strategy': 'Bayesian',
                'HyperParameterTuningJobObjective': {
                    'Type': 'Minimize',
                    'MetricName': 'validation:loss'
                },
                'ResourceLimits': {
                    'MaxNumberOfTrainingJobs': max_jobs,
                    'MaxParallelTrainingJobs': max_parallel_jobs
                },
                'ParameterRanges': {
                    'ContinuousParameterRanges': list(hyperparameter_ranges.values())
                }
            },
            'TrainingJobDefinition': {
                'AlgorithmSpecification': {
                    'TrainingImage': image_uri,
                    'TrainingInputMode': 'File',
                    'MetricDefinitions': [
                        {
                            'Name': 'validation:loss',
                            'Regex': 'eval_loss.*?([0-9\\.]+)'
                        },
                        {
                            'Name': 'train:loss',
                            'Regex': 'train_loss.*?([0-9\\.]+)'
                        }
                    ]
                },
                'RoleArn': self.role_arn,
                'InputDataConfig': [
                    {
                        'ChannelName': 'training',
                        'DataSource': {
                            'S3DataSource': {
                                'S3DataType': 'S3Prefix',
                                'S3Uri': dataset_path,
                                'S3DataDistributionType': 'FullyReplicated'
                            }
                        }
                    }
                ],
                'OutputDataConfig': {
                    'S3OutputPath': f's3://{self.bucket_name}/hp-tuning/{tuning_job_name}/'
                },
                'ResourceConfig': {
                    'InstanceType': instance_type,
                    'InstanceCount': 1,
                    'VolumeSizeInGB': kwargs.get('volume_size', 500)
                },
                'StoppingCondition': {
                    'MaxRuntimeInSeconds': kwargs.get('max_runtime', 14400)  # 4 hours per job
                },
                'StaticHyperParameters': static_hyperparameters
            },
            'Tags': [
                {'Key': 'JobType', 'Value': 'HyperparameterTuning'},
                {'Key': 'Project', 'Value': 'QwenHebrew'},
                {'Key': 'InstanceType', 'Value': instance_type}
            ]
        }
        
        response = self.sagemaker.create_hyper_parameter_tuning_job(**tuning_config)
        logger.info(f"Hyperparameter tuning job {tuning_job_name} submitted")
        return tuning_job_name
    
    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        """Get status and details of a training job"""
        try:
            response = self.sagemaker.describe_training_job(TrainingJobName=job_name)
            return {
                'job_name': job_name,
                'status': response['TrainingJobStatus'],
                'creation_time': response['CreationTime'],
                'training_start_time': response.get('TrainingStartTime'),
                'training_end_time': response.get('TrainingEndTime'),
                'instance_type': response['ResourceConfig']['InstanceType'],
                'failure_reason': response.get('FailureReason')
            }
        except Exception as e:
            logger.error(f"Failed to get status for job {job_name}: {e}")
            return {'job_name': job_name, 'status': 'Unknown', 'error': str(e)}
    
    def wait_for_job_completion(self, job_name: str, check_interval: int = 300) -> Dict[str, Any]:
        """Wait for job completion and return final status"""
        logger.info(f"Waiting for job {job_name} to complete...")
        
        while True:
            status = self.get_job_status(job_name)
            
            if status['status'] in ['Completed', 'Failed', 'Stopped']:
                logger.info(f"Job {job_name} finished with status: {status['status']}")
                return status
            
            logger.info(f"Job {job_name} status: {status['status']}")
            time.sleep(check_interval)
    
    def stop_job(self, job_name: str) -> bool:
        """Stop a running training job"""
        try:
            self.sagemaker.stop_training_job(TrainingJobName=job_name)
            logger.info(f"Stop request sent for job {job_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop job {job_name}: {e}")
            return False
    
    def list_jobs(self, status_filter: Optional[str] = None, max_results: int = 100) -> List[Dict]:
        """List training jobs with optional status filter"""
        try:
            kwargs = {
                'MaxResults': max_results,
                'SortBy': 'CreationTime',
                'SortOrder': 'Descending'
            }
            
            if status_filter:
                kwargs['StatusEquals'] = status_filter
            
            response = self.sagemaker.list_training_jobs(**kwargs)
            return response['TrainingJobSummaries']
        
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return []
    
    def get_job_logs(self, job_name: str) -> str:
        """Get CloudWatch logs for a training job"""
        try:
            logs_client = boto3.client('logs', region_name=self.region)
            
            log_group = '/aws/sagemaker/TrainingJobs'
            log_stream = job_name
            
            response = logs_client.get_log_events(
                logGroupName=log_group,
                logStreamName=log_stream,
                startFromHead=True
            )
            
            log_lines = [event['message'] for event in response['events']]
            return '\n'.join(log_lines)
        
        except Exception as e:
            logger.error(f"Failed to get logs for job {job_name}: {e}")
            return f"Error retrieving logs: {e}"
    
    def download_model_artifacts(self, job_name: str, local_path: str) -> bool:
        """Download trained model artifacts"""
        try:
            # Get job details to find model artifacts location
            job_details = self.sagemaker.describe_training_job(TrainingJobName=job_name)
            model_artifacts_uri = job_details['ModelArtifacts']['S3ModelArtifacts']
            
            # Parse S3 URI
            s3_parts = model_artifacts_uri.replace('s3://', '').split('/', 1)
            bucket = s3_parts[0]
            key = s3_parts[1]
            
            # Download the model.tar.gz file
            import os
            os.makedirs(local_path, exist_ok=True)
            local_file = os.path.join(local_path, 'model.tar.gz')
            
            self.s3.download_file(bucket, key, local_file)
            
            # Extract the tar.gz file
            import tarfile
            with tarfile.open(local_file, 'r:gz') as tar:
                tar.extractall(local_path)
            
            logger.info(f"Model artifacts downloaded to {local_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to download model artifacts for {job_name}: {e}")
            return False

def create_job_manager(role_arn: str, bucket_name: str, region: str = 'us-east-1') -> SageMakerJobManager:
    """Factory function to create SageMaker job manager"""
    return SageMakerJobManager(role_arn, bucket_name, region)

# Example usage functions
def submit_benchmark_jobs(job_manager: SageMakerJobManager, 
                         dataset_path: str,
                         instance_types: List[str] = None) -> List[str]:
    """Submit benchmark jobs across multiple instance types"""
    if instance_types is None:
        instance_types = ['ml.p4d.24xlarge', 'ml.p4de.24xlarge', 'ml.p5.48xlarge']
    
    job_names = []
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    for instance_type in instance_types:
        instance_clean = instance_type.replace('.', '-').replace('ml-', '')
        job_name = f"qwen-benchmark-{instance_clean}-{timestamp}"
        
        job_manager.submit_training_job(
            job_name=job_name,
            instance_type=instance_type,
            dataset_path=dataset_path,
            epochs=1,
            max_steps=100,  # Quick benchmark
            benchmark_mode=True,
            wandb_project='qwen-hebrew-benchmark'
        )
        
        job_names.append(job_name)
        time.sleep(30)  # Avoid throttling
    
    return job_names

def submit_production_training(job_manager: SageMakerJobManager,
                              dataset_path: str,
                              instance_type: str = 'ml.p4de.24xlarge',
                              epochs: int = 3) -> str:
    """Submit production training job"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"qwen-hebrew-production-{timestamp}"
    
    return job_manager.submit_training_job(
        job_name=job_name,
        instance_type=instance_type,
        dataset_path=dataset_path,
        epochs=epochs,
        wandb_project='qwen-hebrew-production',
        checkpoint_s3_uri=f's3://{job_manager.bucket_name}/checkpoints/{job_name}/'
    )