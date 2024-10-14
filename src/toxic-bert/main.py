from pathlib import Path
from config_loader import load_config
from elasticsearch_utils import get_es_source
from text_processing import load_language_detector
from toxicity_analysis import load_toxicity_model, ToxicityClassifier

from ray import init
from ray.data import read_datasource, DataContext
import time


init()
DataContext.get_current().enable_operator_progress_bars = True

def main():
    config_path = Path("data/config/config.yaml")
    config = load_config(config_path)

    batch_size = config['toxicity_analysis']['batch_size']

    es_source = get_es_source(config)

    print("Reading data from Elasticsearch...")
    start_time = time.time()
    read_datasource(es_source) \
    .map_batches(
        ToxicityClassifier(),
        concurrency=100,
        num_gpus=0,  
        batch_size=batch_size) \
    .write_json("local:///mnt/ceph/storage/data-tmp/2024/po87xox/toxicity")

    elapsed_time = time.time() - start_time

    print(f"Time taken: {elapsed_time} seconds")

if __name__ == "__main__":
    main()
