# Textual data extraction from PDF of varying quality

## Overview

This directory is dedicated to the text extraction from PDF. The primary goal is to extract high-quality Hebrew text for Large Language Models. The code in this folder utilizes Vertex batch inference with Gemini for text extraction with tabular data preservation and description placeholders for encountered figures.

## Code

1.  "pdf_batching.py" - script that allows to prepare PDFs located either in AWS or GCP. Steps include splitting the PDF's into pages, validating information presence on a page, conversion to high quality image and uploading to a GCP bucket for future Vertex batching.
2.  "clean_gcp_ocr.ipynb" - notebook with step by step process for writing jsonl file for the batch inference, its uploading and batch invokation. Also includes batch results processing, accumulating text results in the correct order for the original documents.

## Usage

To use:
1. If you have preprocessed your pdf's in a desired way and uploaded them to GCP, you can skip right to the notebook and adjust the jsonl config to your needs.
2. Else, you can start with the pre-processing script that would convert your pdf's into a more optimal format for Gemini.
