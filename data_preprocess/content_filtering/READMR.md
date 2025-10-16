# FineWeb-Based Hebrew Data Filtering Pipeline

This module implements a **FineWeb2 + Gopher-inspired filtering pipeline** for Hebrew datasets.  
It is part of the `qwen-hebrew-finetuning` project and is responsible for filtering raw datasets stored on S3 into high-quality text suitable for model training.

The filtering process is controlled by a single **YAML configuration file**, which defines both **global** filtering parameters and **dataset-specific overrides**.

---

## 1. Overview

The pipeline evaluates datasets according to **quality**, **duplication**, and **linguistic** heuristics.  
Each dataset is processed independently and can override defaults through the YAML configuration.

Core filters used:
- `MinDocFilter` – filters documents with too few words.
- `GopherRepetitionFilter` – detects repeated lines and N-gram duplications.
- `FineWebQualityFilter` – applies FineWeb2-style heuristics (e.g., punctuation and line frequency).
- `GopherQualityFilter` – applies Gopher paper heuristics (word length, stop words, etc.).

Each filter writes both **passed** and **removed** documents to S3, along with **logs** and **Excel summaries**.

---


The stages (with original FineWeb2 thresholds → **our thresholds** in parentheses) are:

1. **Minimum number of words per document**
   - Original: 50 (**ours:** 15)

2. **Gopher duplication measures**
   - Repeated N-gram sequences (character ratio of top N-gram and any N-gram duplication)
   - Repeated lines ratio (character-level and line-level duplication, threshold **0.34**)

3. **FineWeb quality**
   - Character duplication
   - Fraction of lines ending with punctuation (original: 0.2, **ours:** 0.05)  
     *If no line ends with punctuation, the document is **not** filtered out.*
   - Line break frequency (ratio of newlines to total characters)

4. **Gopher quality**
   - Mean word length (**ours:** 2–9)
   - Minimum stop words (original: 2, **ours:** 0–X)
   - Maximum non-alphabetic word ratio (original: 0.8, **ours:** 0.65)

---

## 2. Running the Pipeline

The pipeline can be executed through the main entry point:
```bash
python run_filtering.py
```

The script runs end-to-end filtering for all datasets under the S3 prefix defined in the file constants.

### Required setup:

* AWS credentials configured for S3 access.
* The following dependencies installed:

  ```bash
  pip install datatrove fs spacy pandas s3fs transformers orjson openpyxl pyyaml
  ```

### Key constants in the script:

| Variable         | Description                                                          |
| ---------------- | -------------------------------------------------------------------- |
| `BUCKET_NAME`    | Target S3 bucket containing datasets                                 |
| `INPUT_BASE`     | Base input folder (per-dataset subfolders are scanned automatically) |
| `OUTPUT_BASE`    | Path for filtered outputs and logs                                   |
| `CONFIG_PATH`    | Path to the YAML configuration file (see below)                      |
| `TASKS`          | Number of parallel threads for processing                            |
| `DEBUG_ONE_PART` | If `True`, runs only one input shard per dataset (for testing)       |
| `MAX_ROWS`       | Limit on number of rows read from each file (set to `-1` for all)    |

---

## 3. YAML Configuration File

The YAML file defines **filter parameters** that control all stages of the pipeline.

It contains:

1. **Global defaults** — used for all datasets unless overridden.
2. **Dataset-specific overrides** — customize parameters for a specific dataset.

Example structure:

```yaml
# heb_Hebr.yml

# Global defaults
min_doc_words: 15
max_doc_words: 100000
dup_line_frac: 0.34
top_n_grams: 50
dup_n_grams: 0.15
...

# Per-dataset overrides
dataset_overrides:
  hebrew_tweets:
      min_doc_words: 10
```
See YAML file for full configuration, in that example the min number of word is set for 15, while for twitter is set to 10
---
### Parameter meanings

| Parameter                       | Description                                                          |
| ------------------------------- | -------------------------------------------------------------------- |
| **`min_doc_words`**             | Minimum non-symbol words required per document                       |
| **`max_doc_words`**             | Maximum words allowed (optional)                                     |
| **`dup_line_frac`**             | Fraction threshold for repeated lines                                |
| **`top_n_grams`**               | Number of top N-grams checked for repetition                         |
| **`dup_n_grams`**               | Threshold for N-gram duplication ratio                               |
| **`line_punct_exclude_zero`**   | Whether documents with zero lines ending in punctuation are excluded |
| **`line_punct_thr`**            | Minimum ratio of lines ending with punctuation                       |
| **`new_line_ratio`**            | Ratio of newlines to total characters                                |
| **`max_avg_word_length`**       | Maximum allowed mean word length                                     |
| **`min_avg_word_length`**       | Minimum allowed mean word length                                     |
| **`stopwords`**                 | List of stop words to use (language-specific)                        |
| **`min_number_of_stopwords`**   | Minimum stop words required       - we set it to 0                   |
| **`max_non_alpha_words_ratio`** | Maximum ratio of non-alphabetic tokens (numbers, emojis, etc.)       |
| **`max_bullet_lines_ratio`**    | Max fraction of bullet-like lines                                    |
| **`max_ellipsis_lines_ratio`**  | Max fraction of ellipsis (`...`) lines                               |
| **`dataset_overrides`**         | Dataset-specific overrides of any of the above parameters            |


---

## 4. Outputs

Each dataset produces a structured set of results under:

```
s3://<bucket>/<output_prefix>/<dataset_name>/
```

Contents:

* `filtering/output/` — Filtered (kept) JSONL documents
* `filtering/removed/<filter_name>/` — Rejected samples with reasons
* `logs/` — Per-dataset logs, `stats.json`, Excel summaries, and text reports
* `summary.json` and `overall_summary.json` — Aggregated statistics
* `*_dropped_records_filtered.xlsx` — Excel summary of removed samples and reasons

Each report includes:

* Total and dropped samples
* Drop rate (%)
* Top filtering reasons
* Per-filter statistics

---

## 5. Extending the Configuration

To fine-tune filtering for new datasets:

1. Add a new dataset name under `dataset_overrides`.
2. Override only the parameters you wish to change.
3. Re-run the script — no code modification is required.

Example:

```yaml
dataset_overrides:
  my_new_dataset:
    min_doc_words: 10
    dup_line_frac: 0.4
    max_non_alpha_words_ratio: 0.6
```

---

## 6. Logging and Reports

After completion, the script generates:

* `reason_counts.txt` — detailed breakdown of drop reasons
* `summary.xlsx` — Excel report with rates and per-reason distributions
* `overall_summary.json` — combined statistics across all datasets

---
