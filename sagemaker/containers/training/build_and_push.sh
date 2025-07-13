#!/bin/bash

# Build and push SageMaker training container for Qwen Hebrew fine-tuning

# Configuration
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure list | grep region | awk '{print $2}')
REPOSITORY_NAME="qwen-hebrew-training"
IMAGE_TAG="latest"

# Full image name
FULL_NAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY_NAME}:${IMAGE_TAG}"

echo "Building and pushing Docker image: ${FULL_NAME}"
echo "Account ID: ${ACCOUNT_ID}"
echo "Region: ${REGION}"

# Verify required files exist
echo "Checking required files..."
if [ ! -f "Dockerfile" ]; then
    echo "❌ Error: Dockerfile not found in current directory"
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found in current directory"
    exit 1
fi

if [ ! -f "scripts/train.py" ]; then
    echo "❌ Error: scripts/train.py not found"
    exit 1
fi

echo "✅ All required files found"

# Create ECR repository if it doesn't exist
echo "Checking ECR repository..."
aws ecr describe-repositories --repository-names ${REPOSITORY_NAME} > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Creating ECR repository: ${REPOSITORY_NAME}"
    aws ecr create-repository --repository-name ${REPOSITORY_NAME}
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create ECR repository"
        exit 1
    fi
else
    echo "✅ ECR repository exists"
fi

# Get login token and login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com
if [ $? -ne 0 ]; then
    echo "❌ Failed to login to ECR"
    exit 1
fi
echo "✅ Successfully logged in to ECR"

# Build the Docker image with no cache to ensure we get latest changes
echo "Building Docker image (with --no-cache for latest changes)..."
docker build --no-cache -t ${REPOSITORY_NAME} .
if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi
echo "✅ Docker image built successfully"

# Tag the image
echo "Tagging image..."
docker tag ${REPOSITORY_NAME} ${FULL_NAME}
if [ $? -ne 0 ]; then
    echo "❌ Failed to tag image"
    exit 1
fi
echo "✅ Image tagged successfully"

# Push the image to ECR
echo "Pushing image to ECR..."
docker push ${FULL_NAME}
if [ $? -ne 0 ]; then
    echo "❌ Failed to push image to ECR"
    exit 1
fi

echo ""
echo "🎉 Successfully built and pushed: ${FULL_NAME}"
echo ""
echo "📝 Use this image URI in your SageMaker training jobs:"
echo "   ${FULL_NAME}"
echo ""
echo "🧪 To test the image locally:"
echo "   mkdir -p /tmp/test-data"
echo "   echo '{\"text\": \"Test sentence for training\"}' > /tmp/test-data/train.jsonl"
echo "   docker run --gpus all -it -v /tmp/test-data:/opt/ml/input/data/training ${REPOSITORY_NAME} train --max-steps 5"
echo ""
echo "✅ Build completed successfully!"