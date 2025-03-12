import os
from pathlib import Path
from ray import init
from ray.data import read_datasource
from src.toxic_bert.elasticsearch_utils import get_es_source_deduplication
from src.toxic_bert.config_loader import load_config
from src.toxic_bert.text_processing import extract_text
from toxic_bert.minhash_tools import HashFinder

os.environ["RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S"] = "3600"

init()

def main():
    config_path = Path("data/config/config.yaml")
    config = load_config(config_path)
    output_dir = config["deduplication"]["output_dir"]
    es_source = get_es_source_deduplication(config)

    (
        read_datasource(
            datasource=es_source,
            concurrency=1000,
            override_num_blocks=1000,
            #override_num_blocks=4_000_000,
            ray_remote_args=dict(
                num_cpus=0.01,
                memory=2 * 1024**3,
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
            fn=HashFinder(),
            concurrency=1000,
            num_cpus=0.01,
            batch_format="pandas",
            memory=5 * 1024**3,
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
