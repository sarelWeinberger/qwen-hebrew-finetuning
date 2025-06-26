#!/bin/bash

# Build and push SageMaker training container for Qwen Hebrew fine-tuning

# Configuration
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)
REPOSITORY_NAME="qwen-hebrew-training"
IMAGE_TAG="latest"

# Full image name
FULL_NAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY_NAME}:${IMAGE_TAG}"

echo "Building and pushing Docker image: ${FULL_NAME}"

# Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names ${REPOSITORY_NAME} > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Creating ECR repository: ${REPOSITORY_NAME}"
    aws ecr create-repository --repository-name ${REPOSITORY_NAME}
fi

# Get login token and login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Build the Docker image
echo "Building Docker image..."
docker build -t ${REPOSITORY_NAME} .

# Tag the image
echo "Tagging image..."
docker tag ${REPOSITORY_NAME} ${FULL_NAME}

# Push the image to ECR
echo "Pushing image to ECR..."
docker push ${FULL_NAME}

echo "Successfully built and pushed: ${FULL_NAME}"
echo "Use this image URI in your SageMaker training jobs"