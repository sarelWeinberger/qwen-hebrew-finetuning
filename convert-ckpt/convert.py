import json
import shutil
import os
import logging
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM
from nemo.collections.llm import export_ckpt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Base Paths ---
NEMO_BASE_PATH = Path("/workspace/checkpoints/nemo")
HF_BASE_PATH = Path("/workspace/checkpoints/hf")
# --------------------

def patch_io_json(nemo_checkpoint_path: Path):
    """
    Manually edits the io.json file to prevent hardware and data-loading errors.
    
    This is the core "hack" to allow loading a multi-node checkpoint
    on a single-GPU machine.
    """
    io_config_path = nemo_checkpoint_path / "context" / "io.json"
    log.info(f"Loading configuration from {io_config_path}...")
    
    if not io_config_path.exists():
        log.error(f"FATAL: {io_config_path} not found. Ensure your model exists at {nemo_checkpoint_path}.")
        return False

    # 1. Back up the original file
    backup_path = io_config_path.with_suffix(".json.bak")
    log.info(f"Backing up original config to {backup_path}")
    shutil.copy(io_config_path, backup_path)

    try:
        with open(io_config_path, 'r') as f:
            data = json.load(f)

        # 2. Patch Trainer Config (Fix MisconfigurationException)
        log.info("Patching trainer configuration (devices=1, num_nodes=1)...")
        # Find the trainer config and force devices/nodes to 1
        trainer_items = data["objects"]["trainer_1"]["items"]
        for item in trainer_items:
            if item[0] == "Attr(name='devices')":
                item[1]["value"] = 1
                log.info("  - Set devices = 1")
            if item[0] == "Attr(name='num_nodes')":
                item[1]["value"] = 1
                log.info("  - Set num_nodes = 1")
        
        # 3. Patch Datamodule Config (Fix FileNotFoundError)
        log.info("Patching datamodule configuration (removing data path)...")
        # Find the datamodule reference and empty it to skip data loading
        # This key ('dict_26') is specific to the original training config.
        if "dict_26" in data["objects"]:
             data["objects"]["dict_26"]["items"] = []
             log.info("  - Emptied datamodule items.")
        else:
            log.warning("Could not find 'dict_26' to patch datamodule. This might be okay if config structure is different.")

        # 4. Save the patched file
        with open(io_config_path, 'w') as f:
            json.dump(data, f)
            
        log.info(f"Successfully patched and saved {io_config_path}.")
        return True

    except Exception as e:
        log.error(f"Failed to patch {io_config_path}: {e}", exc_info=True)
        log.error("Restoring from backup...")
        shutil.copy(backup_path, io_config_path)
        return False

def convert_to_safetensors(hf_output_path: Path):
    """
    Loads the converted HF model (.bin) and re-saves it
    using the .safetensors format.
    """
    log.info(f"Loading converted model from {hf_output_path} to create safetensors...")
    
    if not hf_output_path.exists() or not any(hf_output_path.iterdir()):
        log.error(f"Cannot create safetensors. Path {hf_output_path} is empty or does not exist.")
        log.error("This usually means the 'export_ckpt' step failed to create the model files.")
        return

    try:
        model = AutoModelForCausalLM.from_pretrained(hf_output_path)
        log.info("Model loaded. Re-saving with safe_serialization=True...")
        
        model.save_pretrained(hf_output_path, safe_serialization=True)
        
        log.info("Safetensors created.")

        # 5. (Optional) Clean up the old .bin file
        bin_path = hf_output_path / "pytorch_model.bin"
        if bin_path.exists():
            log.info("Cleaning up old pytorch_model.bin file...")
            os.remove(bin_path)
            
    except Exception as e:
        log.error(f"Failed to convert to safetensors: {e}", exc_info=True)
        log.error("The model was converted to Hugging Face format but NOT to safetensors.")

def main():
    parser = argparse.ArgumentParser(description="Convert NeMo checkpoints to Hugging Face safetensors format.")
    parser.add_argument(
        "--nemo_model", 
        type=str, 
        required=True, 
        help="Name of the input NeMo checkpoint directory (relative to /workspace/checkpoints/nemo)"
    )
    parser.add_argument(
        "--hf_model", 
        type=str, 
        required=True, 
        help="Name of the output Hugging Face directory (relative to /workspace/checkpoints/hf)"
    )
    args = parser.parse_args()

    # Construct the full paths
    nemo_checkpoint_path = NEMO_BASE_PATH / args.nemo_model
    hf_output_path = HF_BASE_PATH / args.hf_model
    print(f'======= hf_output_path: {hf_output_path} ============')

    log.info(f"--- Starting Conversion ---")
    log.info(f"Source: {nemo_checkpoint_path}")
    log.info(f"Target: {hf_output_path}")
    
    if not patch_io_json(nemo_checkpoint_path):
        log.error("Could not patch config. Aborting conversion.")
        return

    log.info(f"Starting NeMo to Hugging Face export...")
    try:
        # Call export_ckpt *without* the hf_ref argument, as requested
        export_ckpt(
            path=nemo_checkpoint_path,
            target="hf",
            output_path=hf_output_path
        )
        log.info("NeMo export_ckpt completed successfully.")
        
        # Run the final conversion to safetensors
        convert_to_safetensors(hf_output_path)
        
        log.info(f"--- Conversion Complete! ---")
        log.info(f"Your safetensors model is ready at {hf_output_path}")

    except Exception as e:
        log.error(f"An error occurred during the main conversion: {e}", exc_info=True)
        log.error("Please check the logs. If you saw a 'MisconfigurationException' or 'FileNotFoundError',")
        log.error("the patching logic may need to be updated for your checkpoint's config structure.")

if __name__ == "__main__":
    main()