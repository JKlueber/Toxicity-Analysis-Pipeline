# Thesis Julian Klüber

## Overview

This project is designed to classify text data for toxicity using different models. It integrates various components such as text extraction, language detection, similarity grouping, and toxicity classification. The pipeline is built to handle large datasets efficiently using distributed computing with Ray and Dask.

## Features

- **Text Extraction:** Extracts plaintext from HTML content.
- **Language Detection:** Detects the language of the text using FastText.
- **Similarity Grouping:** Groups similar documents using MinHash and Locality-Sensitive Hashing (LSH).
- **Toxicity Classification:** Classifies text for toxicity using different models:
  - **Detoxify Original:** A pre-trained BERT model for toxicity classification.
  - **Detoxify Unbiased:** A RoBERTa model trained to reduce bias in toxicity classification.
  - **Perspective API:** Uses Perspective API for toxicity analysis.

## Installation

### Clone the repository:
```bash
git clone https://github.com/yourusername/toxicity-classification-pipeline.git
cd toxicity-classification-pipeline
```

### Set up the environment:

#### Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

#### Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Set up environment variables:

Create a `.env` file in the root directory and add your API keys and other configurations:
```plaintext
GOOGLE_API_KEY=your_google_api_key
ELASTICSEARCH_PASSWORD=your_elasticsearch_password
```


## Usage

### Running the Pipeline

To run the pipeline, use the following command:
```bash
ray job submit --runtime-env env.yml --no-wait -- python -m src.toxic_bert.process --model 1
```

#### `--model`: Choose the toxicity classification model:
- `0`: Detoxify Original
- `1`: Detoxify Unbiased
- `2`: Google Perspective API

## Configuration

The pipeline is configured using a YAML file (`config.yaml`). The configuration includes settings for Elasticsearch, date ranges, and output directories.

### Example `config.yaml`:
```yaml
elasticsearch:
  host: your_elasticsearch_host
  port: your_elasticsearch_port
  user: your_elasticsearch_user
  index: your_index_name
  language: "en"

date_range:
  after: <start_time>
  before: <end_time>

toxicity_analysis:
  output_dir: <ceph_output_dir>
```

## Output

The pipeline outputs the results to the specified directory (`output_dir` in the configuration). The output is a Parquet file containing the classified data with toxicity scores for each label.

## Components

### Text Processing
- `text_processing.py`: Extracts plaintext from HTML content and filters out empty or whitespace-only texts.

### Language Detection
- `language_detection.py`: Uses FastText to detect the language of the text. Only English texts are processed further.

### Similarity Grouping
- `similarity_grouping.py`: Groups similar documents using MinHash and LSH to reduce redundancy.

### Toxicity Classification
- `toxicity_classifier_detoxify_original.py`: Implements the Detoxify Original model.
- `toxicity_classifier_detoxify_unbiased.py`: Implements the Detoxify Unbiased model.
- `toxicity_classifier_google.py`: Implements the Perspective API for toxicity classification.

### Dataset Handling
- `dataset.py`: Provides utilities for loading, writing, and merging data. It also includes functions for computing document hashes and finding similar documents.

### Elasticsearch Utilities
- `elasticsearch_utils.py`: Provides functions to connect to Elasticsearch and retrieve data based on the specified configuration.

## Dependencies

- **Ray**: For distributed computing.
- **Dask**: For parallel data processing.
- **FastText**: For language detection.
- **Transformers**: For using pre-trained models (Detoxify).
- **Pandas**: For data manipulation.
- **Elasticsearch**: For data retrieval.

## Acknowledgments

- **Detoxify**: For providing pre-trained models for toxicity classification.
- **Google Perspective API**: For providing an API for toxicity analysis.
- **FastText**: For providing a model for language detection.