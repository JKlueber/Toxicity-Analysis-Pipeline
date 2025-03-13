# Thesis Julian Klüber

## Toxicity Analysis Pipeline

This repository contains a pipeline for analyzing the toxicity of text data using various models. The pipeline is designed to handle large datasets efficiently by leveraging distributed computing with the **Ray framework**. The process involves several steps, including data deduplication, toxicity prediction, and merging the results back into the original dataset.

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Steps](#pipeline-steps)
   - [1. Build LSH (Locality-Sensitive Hashing)](#1-build-lsh-locality-sensitive-hashing)
   - [2. Deduplicate Data](#2-deduplicate-data)
   - [3. Predict Toxicity](#3-predict-toxicity)
   - [4. Merge Data](#4-merge-data)
3. [Running the Pipeline](#running-the-pipeline)
4. [Configuration](#configuration)
5. [Dependencies](#dependencies)
6. [Data Requirements](#data-requirements)

---

## Overview

The pipeline processes text data to predict toxicity using one of three models: **Detoxify Original**, **Detoxify Unbiased**, or **Perspective API**. The pipeline filters English posts and performs deduplication to ensure that the analysis is based on unique data points. The final output is a dataset with predicted toxicity scores merged with the original data.

The pipeline is designed to handle large-scale datasets by distributing the workload across multiple nodes using **Ray**, a framework for distributed computing. This allows for efficient processing of millions of text entries.

---

## Pipeline Steps

### 1. Build LSH (Locality-Sensitive Hashing)

- **File**: `build_lsh.py`
- **Description**: 
  - This step creates a **MinHash LSH (Locality-Sensitive Hashing)** index over the entire dataset. LSH is a technique used to quickly find similar items in large datasets by hashing them into buckets. The MinHash algorithm is used to generate hashes that represent the content of each text entry.
  - The LSH index is built by computing MinHashes for each text entry and inserting them into the LSH structure. This allows for efficient deduplication in the next step.
- **Output**: 
  - A serialized LSH object saved as a `.pkl` file, which is used in the deduplication step.

### 2. Deduplicate Data

- **File**: `deduplicate_data.py`
- **Description**: 
  - This step uses the LSH index created in the previous step to **deduplicate the dataset**. It queries the LSH index to find duplicate entries and removes them, ensuring that only unique text entries are processed further.
  - The deduplication process is based on the MinHash values computed for each text entry. If two entries have similar MinHash values, they are considered duplicates.
- **Output**: 
  - A deduplicated dataset saved in **Parquet format**, which is used as input for the toxicity prediction step.

### 3. Predict Toxicity

- **File**: `predict_data_toxicity.py`
- **Description**: 
  - This step predicts the toxicity of the deduplicated dataset using one of the three available models:
    1. **Detoxify Original**: A model trained on the Jigsaw Unintended Bias in Toxicity Classification dataset.
    2. **Detoxify Unbiased**: A model trained to reduce bias in toxicity predictions.
    3. **Perspective API**: A model provided by Google that uses machine learning to identify toxic content.
  - Before predicting toxicity, the pipeline filters out non-English posts using the **FastText language detection model**.
- **Output**: 
  - A dataset with predicted toxicity scores saved in **Parquet format**.

### 4. Merge Data

- **File**: `merge_data.py`
- **Description**: 
  - This step merges the predicted toxicity scores with the original dataset. It uses the LSH index to match the deduplicated entries with their corresponding toxicity scores.
  - The final dataset contains the original text data along with the predicted toxicity scores, allowing for further analysis.
- **Output**: 
  - A final dataset with predicted toxicity scores merged with the original data, saved in **Parquet format**.

---

## Running the Pipeline

To run the pipeline, you need to execute the steps in the following order:

1. **Build LSH**:
```bash
ray job submit --runtime-env env.yml --no-wait -- python -m src.deduplication.build_lsh
```

2. **Deduplicate Data**:
```bash
ray job submit --runtime-env env.yml --no-wait -- python -m src.deduplication.deduplicate_data
```

3. **Predict Toxicity**:
```bash
ray job submit --runtime-env env.yml --no-wait -- python -m src.toxicity_predicting.predict_data_toxicity --model <model_number>
```
Replace `<model_number>` with:
- `0` for **Detoxify Original**
- `1` for **Detoxify Unbiased**
- `2` for **Perspective API**

---

Replace `<model_number>` with `0` for Detoxify Original, `1` for Detoxify Unbiased, or `2` for Perspective API.

4. **Merge Data**:
```bash
ray job submit --runtime-env env.yml --no-wait -- python -m src.merging.merge_data
```

## Configuration

The pipeline is configured using a `config.yaml` file and a `.env` file. Below are the details of these configuration files:

### `config.yaml`

The `config.yaml` file contains the following sections:

```bash
elasticsearch:
  host: "your_elasticsearch_host_url"
  port: 9200
  user: "your_elasticsearch_username"
  index: "your_elasticsearch_index_pattern"
  language: "en"
  instances_file: "path/to/instances_file.txt"

date_range:
  after: "2024-01-01T00:00:00"
  before: "2024-12-31T23:59:59"

toxicity_analysis:
  output_dir: "path/to/toxicity_analysis_output"

deduplication:
  output_dir: "path/to/deduplication_output"

merge:
  output_dir: "path/to/merge_output"
```

Make sure to replace the placeholders with your actual configuration values:
- `your_elasticsearch_host_url`: The URL of your Elasticsearch host.
- `your_elasticsearch_username`: The username for Elasticsearch.
- `your_elasticsearch_index_pattern`: The index pattern for your Elasticsearch data.
- `path/to/instances_file.txt`: The path to the file containing the list of Mastodon instances.
- `2024-01-01T00:00:00` and `2024-12-31T23:59:59`: The date range for the data you want to process.
- `path/to/toxicity_analysis_output`: The directory where the toxicity analysis results will be saved.
- `path/to/deduplication_output`: The directory where the deduplicated data will be saved.
- `path/to/merge_output`: The directory where the final merged data will be saved.

### `.env`

The `.env` file is used to store sensitive information such as the Elasticsearch password and the Google API key. It should contain the following variables:

```plaintext
ELASTICSEARCH_PASSWORD=your_elasticsearch_password_here
GOOGLE_API_KEY=your_google_api_key_here
```

Make sure to replace:
- `your_elasticsearch_password_here` with the actual password for the Elasticsearch user specified in the `config.yaml`.
- `your_google_api_key_here` with your Google API key, which is required for the Perspective API toxicity model.

---

## Dependencies

The pipeline requires several Python packages, which can be installed using the `requirements.txt` file. To install the dependencies, run:

CODE
pip install -r requirements.txt
CODE

The main dependencies include:
- **Ray**: For distributed computing.
- **Pandas**: For data manipulation.
- **Datasketch**: For MinHash and LSH operations.
- **FastText**: For language detection.
- **Hugging Face Transformers**: For the Detoxify models.
- **Elasticsearch**: For querying the Elasticsearch database.

These dependencies are also listed in the `env.yml` file, which is used to create the runtime environment for the Ray jobs.

---

## Data Requirements

To use this pipeline, your data in Elasticsearch must have at least the following columns:
- `_id`: A unique identifier for each post.
- `content`: The HTML text or plaintext of the posts.

For any additional metadata you have (e.g., timestamps, user information, etc.), you can modify the Elasticsearch query in the `merge_data.py` step to include these fields. This allows you to merge the predicted toxicity scores with the original data, including any additional metadata you want to retain.

---

## Additional Notes

- **Ray Framework**: The pipeline leverages Ray for distributed computing, allowing it to scale across multiple nodes and handle large datasets efficiently.
- **MinHash and LSH**: These techniques are used for efficient deduplication, ensuring that the toxicity analysis is performed on unique data points.
- **Language Detection**: The pipeline uses FastText to filter out non-English posts, ensuring that the toxicity analysis is focused on English text.

By following the steps outlined in this README, you can efficiently analyze the toxicity of large text datasets using distributed computing and state-of-the-art toxicity prediction models.