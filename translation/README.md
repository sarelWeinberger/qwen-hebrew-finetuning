# Translation Folder README

## Overview

This directory is dedicated to the translation of various English benchmarks into Hebrew. The primary goal is to create high-quality Hebrew benchmarks for Large Language Models. The notebooks in this folder utilize Gemini for translation and provide tools for evaluation and labeling.

## Directory Structure

*   **`/plots`**: This directory contains various plots and visualizations generated during the analysis and evaluation of the translations. These include JPEG images that show rating comparisons, distributions of MQM (Machine Quality Metric) scores for benchmarks like MMLU, ARC, and GSM, and confusion matrices for rating overlaps.
*   **`/prompts`**: This directory stores Python files (`.py`) that define the prompt templates used for guiding the translation model. Each file is named after the benchmark it corresponds to (e.g., `mmlu_prompts.py`, `hellaswag_prompts.py`, `arc_prompts.py`) and contains the specific instructions fed to the model to ensure accurate and consistent translation for that benchmark's format.
*   **`/src`**: This directory holds source code and utility functions used across the different notebooks. It contains several Python scripts, such as `call_models.py` for interacting with the translation models, `translate_func.py` which holds the core translation logic, `parse_labeling.py` for processing evaluation data, and `gradio_utils.py` for building the labeling application. It also includes subdirectories for more specific code related to benchmarks and instruction data.


## Notebooks

The notebooks are numbered to indicate a suggested order of execution and workflow:

1.  **`1 - translate_funcs.ipynb`**: This notebook contains the core functions for translating text from English to Hebrew.
2.  **`1.1 - translate_funcs_mmlu.ipynb`**: A specialized notebook for translating the MMLU (Massive Multitask Language Understanding) dataset.
3.  **`1.2 - translate_funcs_hellaswag.ipynb`**: A notebook specifically for translating the Hellaswag dataset.
4.  **`2 - labeling_app.ipynb`**: An interactive application for manual labeling and correction of the translations.
5.  **`3 - Evaluate labeling.ipynb`**: This notebook provides tools and methods for evaluating the quality of the translated labels.
6.  **`4 - final hebrew benchmarks.ipynb`**: This notebook is used for creating the final Hebrew benchmarks with a .jsonl format.

## Usage

To use the resources in this folder, it is recommended to follow the notebooks in their numerical order. Start with the `1 - translate_funcs.ipynb` notebooks to perform the translations, then use the evaluation and labeling notebooks to assess and refine the quality of the translated data. Finally, run the benchmark notebook to create the final Hebrew benchmarks.