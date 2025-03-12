import os
from pathlib import Path
from ray import init
from ray.data import read_datasource

from src.toxic_bert.elasticsearch_utils import get_es_source_deduplication
from src.toxic_bert.config_loader import load_config
from src.toxic_bert.text_processing import extract_text
from toxic_bert.minhash_tools import LSHBuilder

import pandas as pd
from datasketch import MinHashLSH

import pickle

from ray.data.aggregate import AggregateFn

os.environ["RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S"] = "3600"

init()

def main():
    config_path = Path("data/config/config.yaml")
    config = load_config(config_path)
    output_dir = config["deduplication"]["output_dir"]
    es_source = get_es_source_deduplication(config)

    lsh_ds = (
        read_datasource(
            datasource=es_source,
            concurrency=1000,
            override_num_blocks=400000,
            #override_num_blocks=4_000_000,
            ray_remote_args=dict(
                num_cpus=0.01,
                memory=2 * 1024**3,
                max_retries=10, 
                retry_exceptions=True,
            ),
        )
        .map_batches(
            fn=extract_text,
            concurrency=1000,
            num_cpus=0.01,
            batch_format="pandas",
            memory=2 * 1024**3,
        )
        .map_batches(
            fn=LSHBuilder(),
            concurrency=1000,
            num_cpus=0.01,
            batch_format="pandas",
            memory=2 * 1024**3,
        ) 
        .materialize()
    )

    def merge_lsh(lsh1: MinHashLSH, lsh2: MinHashLSH) -> MinHashLSH:
        lsh1.merge(lsh2)
        return lsh1
    
    def accumulate_lsh(lsh: MinHashLSH, row: pd.DataFrame) -> MinHashLSH:
        lsh.merge(row["lsh"])
        return lsh
    
    aggregation = AggregateFn(
        init=lambda _: MinHashLSH(num_perm=128, threshold=0.9),
        accumulate_row=accumulate_lsh,
        merge = merge_lsh,
        name="lsh"
    )

    lsh_dict = lsh_ds.aggregate(aggregation)
    file_name = "lsh_test.pkl"
    lsh = lsh_dict["lsh"]

    with open(output_dir+file_name, "wb") as file:
        pickle.dump(lsh, file)

if __name__ == "__main__":
    main()
