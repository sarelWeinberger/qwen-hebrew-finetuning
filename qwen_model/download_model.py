import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model():
    """
    Download the Qwen/Qwen3-30B-A3B-Base model from Hugging Face
    """
    print("Starting model download...")
    
    # Set the model path
    model_path = os.path.join(os.getcwd(), "qwen_model/model")
    
    # Download the model files
    snapshot_download(
        repo_id="Qwen/Qwen3-30B-A3B-Base",
        local_dir=model_path,
        ignore_patterns=["*.bin", "*.safetensors"],  # First download only the config files
        local_dir_use_symlinks=False
    )
    
    print(f"Model config files downloaded to {model_path}")
    print("Now downloading model weights (this may take a while)...")
    
    # Download the model weights
    snapshot_download(
        repo_id="Qwen/Qwen3-30B-A3B-Base",
        local_dir=model_path,
        ignore_patterns=["*.md", "*.txt"],  # Now download the model weights
        resume_download=True,
        local_dir_use_symlinks=False
    )
    
    print(f"Model weights downloaded to {model_path}")
    
    # Load and test the tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
    
    print("Download complete!")

if __name__ == "__main__":
    download_model()