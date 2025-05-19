# Running Training Commands with Nohup

These commands will keep your training processes running even if your SSH connection is lost or you close your terminal.

## Training with 8 H100 GPUs

```bash
# Make the script executable first
chmod +x qwen_model/run_training.sh

# Run the training script with nohup
nohup torchrun --nproc_per_node=8 qwen_model/train.py \
  --dataset_path qwen_model/data/dataset/dataset \
  --config qwen_model/finetuning/training_config.json \
  --deepspeed qwen_model/finetuning/h100_ds_config.json \
  > qwen_model/logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Check the process ID
echo $! > qwen_model/logs/train_pid.txt
echo "Training process started with PID: $!"
echo "To monitor progress: tail -f qwen_model/logs/train_*.log"
```

## Hyperparameter Tuning with Optuna

```bash
# Create logs directory if it doesn't exist
mkdir -p qwen_model/logs

# Run hyperparameter tuning with nohup
nohup python qwen_model/hp_tuning.py \
  --dataset_path qwen_model/data/dataset/dataset \
  --output_dir qwen_model/finetuning/hp_tuning \
  --num_trials 10 \
  > qwen_model/logs/hp_tune_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Check the process ID
echo $! > qwen_model/logs/hp_tune_pid.txt
echo "Hyperparameter tuning process started with PID: $!"
echo "To monitor progress: tail -f qwen_model/logs/hp_tune_*.log"
```

## Evaluation on Hebrew LLM Leaderboard

```bash
# Create logs directory if it doesn't exist
mkdir -p qwen_model/logs

# Run evaluation with nohup
nohup python qwen_model/evaluate_hebrew.py \
  --model_path qwen_model/finetuned \
  --dataset_path qwen_model/data/evaluation_dataset \
  > qwen_model/logs/evaluate_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Check the process ID
echo $! > qwen_model/logs/evaluate_pid.txt
echo "Evaluation process started with PID: $!"
echo "To monitor progress: tail -f qwen_model/logs/evaluate_*.log"
```

## Alternative: Using tmux

If you prefer using tmux instead of nohup, you can use these commands:

```bash
# Install tmux if not already installed
# sudo apt-get install tmux

# Create a new tmux session
tmux new-session -d -s qwen_training

# Send the training command to the tmux session
tmux send-keys -t qwen_training "torchrun --nproc_per_node=8 qwen_model/train.py --dataset_path qwen_model/data/dataset/dataset --config qwen_model/finetuning/training_config.json --deepspeed qwen_model/finetuning/h100_ds_config.json" C-m

# To attach to the tmux session later
# tmux attach-session -t qwen_training

# To detach from the session (without killing it)
# Press Ctrl+b, then d
```

## Monitoring and Managing Background Processes

```bash
# List all running processes
ps aux | grep python

# Monitor log files
tail -f qwen_model/logs/train_*.log

# Kill a process by PID
kill -9 $(cat qwen_model/logs/train_pid.txt)