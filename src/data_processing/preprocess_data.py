import pandas as pd

import os
from ray import init
from ray.data import read_parquet_bulk
from pathlib import Path

from src.config.config_loader import load_config
from src.utils.minhash_utils import CalculateMinHash
from src.data_processing.text_extractor import extract_text

os.environ["RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S"] = "3600"

init(runtime_env={"config": {"setup_timeout_seconds": 3600}})

def sample_batch(batch: pd.DataFrame) -> pd.DataFrame:
    return batch.sample(frac=0.1, random_state=42)

def main():
    config = load_config()
    input_dir = config["dir"]["input_dir"]
    output_dir = config["dir"]["input_sample_dir"]

    file_paths = [str(file) for file in Path(input_dir).glob("*.parquet")]

    (
        read_parquet_bulk(
            paths=file_paths,
            concurrency=100,
            ray_remote_args=dict(
                num_cpus=0.01,
                memory=4 * 1024**3,
                max_retries=10, 
                retry_exceptions=True,
            )
        )
        .map_batches(
            fn=sample_batch,
            concurrency=100,
            num_cpus=0.01,
            batch_format="pandas",
            memory=4 * 1024**3,
            max_retries=10, 
            retry_exceptions=True,
        )
        .map_batches(
            fn=extract_text,
            concurrency=100,
            num_cpus=0.01,
            batch_format="pandas",
            memory=10 * 1024**3,
            max_retries=10, 
            retry_exceptions=True,
        )
        .map_batches(
            fn=CalculateMinHash(),
            concurrency=100,
            num_cpus=0.01,
            batch_format="pandas",
            memory=10 * 1024**3,
            max_retries=10, 
            retry_exceptions=True,
        )
        .write_parquet(
            path=output_dir,
            concurrency=100,
            min_rows_per_file = 100_000,
            ray_remote_args=dict(
                num_cpus=0.01,
                memory=4 * 1024**3,
                max_retries=10, 
                retry_exceptions=True,
            )
        )
    )


if __name__ == "__main__":
    main()
