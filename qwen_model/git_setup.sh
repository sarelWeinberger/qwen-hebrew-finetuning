#!/bin/bash

# Script to initialize a Git repository, add all files, and push to a remote repository

# Default values
GIT_REPO_URL=""
BRANCH_NAME="main"
COMMIT_MESSAGE="Initial commit with Qwen Hebrew fine-tuning code"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --repo)
      GIT_REPO_URL="$2"
      shift
      shift
      ;;
    --branch)
      BRANCH_NAME="$2"
      shift
      shift
      ;;
    --message)
      COMMIT_MESSAGE="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [ -z "$GIT_REPO_URL" ]; then
  echo "Error: --repo is required"
  echo "Usage: ./qwen_model/git_setup.sh --repo <git_repo_url> [--branch <branch_name>] [--message <commit_message>]"
  exit 1
fi

# Create .gitignore file
echo "Creating .gitignore file..."
cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Model files
qwen_model/model/
qwen_model/finetuned/
qwen_model/test_run/
qwen_model/finetuning/hp_tuning/

# Data files
qwen_model/data/raw/
qwen_model/data/dataset/

# Logs
qwen_model/logs/
wandb/
lightning_logs/
runs/

# Environment
.env
.venv
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
EOL

# Initialize Git repository
echo "Initializing Git repository..."
git init

# Add all files
echo "Adding files to Git..."
git add .

# Commit changes
echo "Committing changes..."
git commit -m "$COMMIT_MESSAGE"

# Add remote repository
echo "Adding remote repository..."
git remote add origin $GIT_REPO_URL

# Create and switch to branch
echo "Creating and switching to branch: $BRANCH_NAME..."
git branch -M $BRANCH_NAME

# Push to remote repository
echo "Pushing to remote repository..."
git push -u origin $BRANCH_NAME

echo "Git setup complete!"
echo "Repository: $GIT_REPO_URL"
echo "Branch: $BRANCH_NAME"
echo "Commit message: $COMMIT_MESSAGE"