import json
import shutil
import os
import logging
import argparse
from pathlib import Path
# Note: We no longer need AutoModelForCausalLM from transformers here
from nemo.collections.llm import export_ckpt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Base Paths ---
# REMOVED NEMO_BASE_PATH constant
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
        log.error(f"FATAL: {io_config_path} not found. Ensure the path is correct.")
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
        if "trainer_1" in data["objects"]:
            trainer_items = data["objects"]["trainer_1"]["items"]
            for item in trainer_items:
                if item[0] == "Attr(name='devices')":
                    item[1]["value"] = 1
                    log.info("  - Set devices = 1")
                if item[0] == "Attr(name='num_nodes')":
                    item[1]["value"] = 1
                    log.info("  - Set num_nodes = 1")
        else:
             log.warning("Could not find 'trainer_1' object to patch devices/nodes.")


        # 3. Patch Datamodule Config (Fix FileNotFoundError)
        log.info("Patching datamodule configuration (removing data path)...")
        if "dict_26" in data["objects"]:
             data["objects"]["dict_26"]["items"] = []
             log.info("  - Emptied datamodule items.")
        else:
            log.warning("Could not find 'dict_26' to patch datamodule.")

        # 4. Patch Model Precision to bfloat16
        log.info("Patching model precision to bfloat16...")
        bf16_type_ref = {"type": "pyref", "module": "torch", "name": "bfloat16"}

        model_config_key = "qwen3_config8_b_1"
        if model_config_key in data["objects"]:
            model_config_items = data["objects"][model_config_key]["items"]
            precision_patched = False
            for item in model_config_items:
                if item[0] in ["Attr(name='params_dtype')", "Attr(name='autocast_dtype')", "Attr(name='pipeline_dtype')"]:
                    item[1] = bf16_type_ref
                    log.info(f"  - Set {item[0].split("'")[1]} = bfloat16")
                    precision_patched = True
                if item[0] == "Attr(name='bf16')":
                     item[1]["value"] = True
                     log.info("  - Set bf16 = True")
                if item[0] == "Attr(name='fp16')":
                     item[1]["value"] = False
                     log.info("  - Set fp16 = False")
            if not precision_patched:
                 log.warning(f"Did not find specific dtype settings (params_dtype, etc.) in {model_config_key}. Export might default to another precision.")
        else:
            log.warning(f"Could not find model config object '{model_config_key}' to patch precision.")

        # 5. Save the patched file
        with open(io_config_path, 'w') as f:
            json.dump(data, f)

        log.info(f"Successfully patched and saved {io_config_path}.")
        return True

    except Exception as e:
        log.error(f"Failed to patch {io_config_path}: {e}", exc_info=True)
        log.error("Restoring from backup...")
        shutil.copy(backup_path, io_config_path)
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert NeMo checkpoints to Hugging Face safetensors format.")
    # --- CHANGED LINE 1 ---
    # Updated help text for --nemo_model
    parser.add_argument(
        "--nemo_model",
        type=str,
        required=True,
        help="Full path to the input NeMo checkpoint directory"
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        required=True,
        help="Name of the output Hugging Face directory (relative to /workspace/checkpoints/hf)"
    )
    args = parser.parse_args()

    nemo_checkpoint_path = Path(args.nemo_model)
    hf_output_path = HF_BASE_PATH / args.hf_model
    print(f'======= hf_output_path: {hf_output_path} ============') # Keep your debug print

    log.info(f"--- Starting Conversion ---")
    log.info(f"Source: {nemo_checkpoint_path}")
    log.info(f"Target: {hf_output_path}")

    if not patch_io_json(nemo_checkpoint_path):
        log.error("Could not patch config. Aborting conversion.")
        return

    log.info(f"Starting NeMo to Hugging Face export (precision set in config)...")
    try:
        export_ckpt(
            path=nemo_checkpoint_path,
            target="hf",
            output_path=hf_output_path
        )
        log.info("NeMo export_ckpt completed successfully.")

        log.info(f"--- Conversion Complete! ---")
        log.info(f"Your bfloat16 safetensors model should be ready at {hf_output_path}")

    except Exception as e:
        log.error(f"An error occurred during the main conversion: {e}", exc_info=True)
        log.error("Please check the logs. If you saw a 'MisconfigurationException' or 'FileNotFoundError',")
        log.error("the patching logic may need to be updated for your checkpoint's config structure.")

if __name__ == "__main__":
    main()
