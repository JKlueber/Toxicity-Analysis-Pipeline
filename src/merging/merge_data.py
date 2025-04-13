import os
from pathlib import Path

from ray import init
from ray.data import read_parquet_bulk

from src.config.config_loader import load_config
from src.utils.minhash_utils import MergeHash

os.environ["RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S"] = "3600"

init(ignore_reinit_error=True)

def main():
    config = load_config()
    input_dir = config["dir"]["input_sample_dir"]
    output_dir = config["dir"]["merge_dir"]
    file_paths = [str(file) for file in Path(input_dir).glob("*.parquet")]

    (
        read_parquet_bulk(
            paths=file_paths,
            concurrency=100,
            ray_remote_args=dict(
                num_cpus=0.01,
                memory=2 * 1024**3,
                max_retries=10, 
                retry_exceptions=True,
            )
        )
        .map_batches(
            fn=MergeHash(),
            concurrency=50,
            num_cpus=0.01,
            batch_format="pandas",
            memory=20 * 1024**3,
            max_retries=10, 
            retry_exceptions=True,
        )
        .write_parquet(
            path=output_dir,
            concurrency=100,
            ray_remote_args=dict(
                num_cpus=0.01,
                memory=2 * 1024**3,
                max_retries=10, 
                retry_exceptions=True,
            )
        )
    )

if __name__ == "__main__":
    main()
