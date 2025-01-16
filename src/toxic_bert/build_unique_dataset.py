from pathlib import Path

from ray import init
from ray.data import read_datasource

from src.toxic_bert.config_loader import load_config
from src.toxic_bert.elasticsearch_utils import get_es_source
import pandas as pd

init()

def remove_dublicates(batch: pd.DataFrame) -> pd.DataFrame:
    unique_batch = batch.drop_duplicates(subset=["content"], keep="first")
    return unique_batch

def main():
    config_path = Path("data/config/config.yaml")
    config = load_config(config_path)

    es_source = get_es_source(config)


    (
        read_datasource(
            datasource=es_source,
            concurrency=100,
            override_num_blocks=1000,
            ray_remote_args=dict(
                num_cpus=0.01,
            ),
        )
        .map_batches(remove_dublicates, batch_format="pandas")
        .write_json(
            path="/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-klueber/toxicity",
            concurrency=100,
            ray_remote_args=dict(
                num_cpus=0.01,
            ),
        )
    )

if __name__ == "__main__":
    main()
