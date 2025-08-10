import optuna
import json
import subprocess
import re

CONFIG_PATH = "cpt_config.json"
DS_CONFIG_PATH = "deepspeed_zero3.yaml"
TRAIN_CMD_BASE = [
    "accelerate", "launch", "--config_file=deepspeed_zero3.yaml",
    "train.py", "--wandb_name", "optuna-qwen-hebrew"
]

# Helper to update config files for each trial
def update_config(learning_rate, warmup_ratio, grad_acc_steps, max_steps=100,
                  weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    config["learning_rate"] = learning_rate
    config["warmup_ratio"] = warmup_ratio
    config["gradient_accumulation_steps"] = grad_acc_steps
    config["max_steps"] = max_steps
    config["weight_decay"] = weight_decay
    config["adam_beta1"] = adam_beta1
    config["adam_beta2"] = adam_beta2
    config["adam_epsilon"] = adam_epsilon
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    # Also update deepspeed config for gradient_accumulation_steps
    with open(DS_CONFIG_PATH, "r") as f:
        lines = f.readlines()
    with open(DS_CONFIG_PATH, "w") as f:
        for line in lines:
            if "gradient_accumulation_steps:" in line:
                f.write(f"  gradient_accumulation_steps: {grad_acc_steps}\n")
            else:
                f.write(line)

# Helper to run training and extract train-loss
def run_training_and_get_loss(adam_beta1, adam_beta2, adam_epsilon, weight_decay):
    cmd = TRAIN_CMD_BASE + [
        "--adam_beta1", str(adam_beta1),
        "--adam_beta2", str(adam_beta2),
        "--adam_epsilon", str(adam_epsilon),
        "--weight_decay", str(weight_decay)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    # Try to extract the last train-loss from output (adapt regex if needed)
    # Looks for lines like: {'loss': 1.9962, ...}
    losses = re.findall(r"'loss': ([0-9.]+)", output)
    if losses:
        return float(losses[-1])
    else:
        return float("inf")

# Optuna objective function
def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
    warmup_ratio = trial.suggest_uniform('warmup_ratio', 0.01, 0.1)
    grad_acc_steps = trial.suggest_categorical('gradient_accumulation_steps', [1, 2, 4, 8])
    weight_decay = trial.suggest_uniform('weight_decay', 0.0, 0.1)
    adam_beta1 = trial.suggest_uniform('adam_beta1', 0.85, 0.95)
    adam_beta2 = trial.suggest_uniform('adam_beta2', 0.98, 0.999)
    adam_epsilon = trial.suggest_loguniform('adam_epsilon', 1e-8, 1e-6)
    update_config(learning_rate, warmup_ratio, grad_acc_steps, max_steps=100,
                  weight_decay=weight_decay, adam_beta1=adam_beta1,
                  adam_beta2=adam_beta2, adam_epsilon=adam_epsilon)
    train_loss = run_training_and_get_loss(adam_beta1, adam_beta2, adam_epsilon, weight_decay)
    return train_loss

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    print("Best trial:", study.best_trial)
