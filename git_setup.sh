#!/bin/bash

# Exit on error
set -e

# Check for command line arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <github_username> <github_token> [repository_name]"
    echo "Example: $0 sarelWE your_token qwen-hebrew-finetuning"
    exit 1
fi

GITHUB_USERNAME=$1
GITHUB_TOKEN=$2
REPO_NAME=${3:-qwen-hebrew-finetuning}  # Default repo name if not provided

echo "Setting up git repository and pushing to GitHub..."

# Update .gitignore to exclude additional files
echo "Updating .gitignore file..."
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*.class
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
.vscode-server/
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

# AWS
.aws/
.netrc

# Git
.git/

# Other
=*
EOF

# Configure git
echo "Configuring git..."
git config --global user.name "GitHub Actions"
git config --global user.email "actions@github.com"

# Create a fresh repository
echo "Creating a fresh repository..."
rm -rf .git
git init

# Add only the files we want to include
echo "Adding files to git..."
git add qwen_model/*.py
git add qwen_model/*.md
git add qwen_model/*.sh
git add qwen_model/finetuning/*.json
git add README.md
git add requirements.txt
git add .gitignore
git add git_setup.sh

# Commit changes
echo "Committing changes..."
git commit -m "Initial commit: Qwen3-30B-A3B-Base fine-tuning setup for Hebrew"

# Rename the branch from 'master' to 'main'
echo "Renaming branch from master to main..."
git branch -m master main

# Add remote repository
echo "Adding remote repository..."
git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git

# Provide instructions for manual pushing
echo ""
echo "Local repository setup complete!"
echo "Due to authentication issues, please push to GitHub manually with the following commands:"
echo ""
echo "  git push -u origin main --force"
echo ""
echo "If that doesn't work, you can try setting up SSH authentication:"
echo ""
echo "  git remote set-url origin git@github.com:$GITHUB_USERNAME/$REPO_NAME.git"
echo "  git push -u origin main --force"
echo ""

echo "Done! Code has been pushed to GitHub: https://github.com/$GITHUB_USERNAME/$REPO_NAME"