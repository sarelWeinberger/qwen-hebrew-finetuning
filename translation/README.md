# Translation Folder README

## Overview

This directory is dedicated to the translation of various English datasets into Hebrew. The primary goal is to create high-quality Hebrew datasets for training and benchmarking Large Language Models. The notebooks in this folder utilize a finetuned Qwen model for translation and provide tools for evaluation and labeling.

## Directory Structure

*   **`/plots`**: This directory contains various plots generated during the analysis and evaluation of the translations.
*   **`/prompts`**: This directory stores different prompt templates used for guiding the translation model.
*   **`/src`**: This directory holds source code and utility functions used across the different notebooks.

## Notebooks

The notebooks are numbered to indicate a suggested order of execution and workflow:

1.  **`1 - translate_funcs.ipynb`**: This notebook contains the core functions for translating text from English to Hebrew.
2.  **`1.1 - translate_funcs_mmlu.ipynb`**: A specialized notebook for translating the MMLU (Massive Multitask Language Understanding) dataset.
3.  **`1.2 - translate_funcs_hellaswag.ipynb`**: A notebook specifically for translating the Hellaswag dataset.
4.  **`3 - Evaluate labeling.ipynb`**: This notebook provides tools and methods for evaluating the quality of the translated labels.
5.  **`4 - labeling_app.ipynb`**: An interactive application for manual labeling and correction of the translations.
6.  **`5 - final hebrew benchmarks.ipynb`**: This notebook is used for running the final benchmarks on the newly created Hebrew datasets.
7.  **`6 - translate_instruct.ipynb`**: A notebook focused on translating instruction-based datasets.

## Usage

To use the resources in this folder, it is recommended to follow the notebooks in their numerical order. Start with the `translate_funcs.ipynb` notebooks to perform the translations, then use the evaluation and labeling notebooks to assess and refine the quality of the translated data. Finally, run the benchmark notebook to measure the performance on the Hebrew datasets.