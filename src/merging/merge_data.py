import os
from pathlib import Path

from ray import init
from ray.data import read_datasource

from utils.elasticsearch_utils import get_es_source
from config.config_loader import load_config
from data_processing.text_extractor import extract_text
from utils.minhash_utils import MergeHash

os.environ["RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S"] = "3600"

init(ignore_reinit_error=True)

def main():
    config_path = Path("config/config.yaml")
    config = load_config(config_path)
    output_dir = config["merge"]["output_dir"]
    es_source = get_es_source(config)

    (
        read_datasource(
            datasource=es_source,
            concurrency=1000,
            override_num_blocks=1000,
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
            fn=MergeHash(),
            concurrency=100,
            num_cpus=0.5,
            batch_format="pandas",
            memory=20 * 1024**3,
        )
        .write_parquet(
            path=output_dir,
            ray_remote_args=dict(
                num_cpus=0.01,
                memory=2 * 1024**3,
            )
        )
    )

if __name__ == "__main__":
    main()
