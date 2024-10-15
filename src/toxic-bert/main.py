from pathlib import Path
from config_loader import load_config
from elasticsearch_utils import get_es_source
from toxicity_analysis import ToxicityClassifier

from ray import init
from ray.data import read_datasource, DataContext

init()
DataContext.get_current().enable_operator_progress_bars = True
RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=1

def main():
    config_path = Path("data/config/config.yaml")
    config = load_config(config_path)

    batch_size = config['toxicity_analysis']['batch_size']

    es_source = get_es_source(config)

    print("Reading data from Elasticsearch...")
    read_datasource(es_source, concurrency=40) \
    .map_batches(
        ToxicityClassifier(),
        concurrency=100,
        num_gpus=0,  
        batch_size=batch_size) \
    .write_json("local:///mnt/ceph/storage/data-tmp/2024/po87xox/toxicity")
    
if __name__ == "__main__":
    main()
