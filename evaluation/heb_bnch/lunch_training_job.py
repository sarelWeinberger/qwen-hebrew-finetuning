from sagemaker.estimator import Estimator

image_uri = '670967753077.dkr.ecr.us-east-1.amazonaws.com/benchmark-runner:latest'
role = 'arn:aws:iam::670967753077:role/SageMakerLightEvalRole'

estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type='ml.g5.2xlarge',
    volume_size=400,
    max_run=86400,
    environment={
        'MODEL_SOURCE': 'huggingface',
        'MODEL_NAME': 'CohereLabs/aya-expanse-8b',
        'HF_TOKEN': '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',
        'BACKEND': 'vllm',
        'MAX_SAMPLES': '3000',
        'GIT_PYTHON_REFRESH': 'quiet'
    },
    output_path='s3://gepeta-datasets/benchmark_results/'
)


estimator.fit({
    'code': 's3://gepeta-datasets/lighteval/custom_tasks_new_version/'
}, wait=False)