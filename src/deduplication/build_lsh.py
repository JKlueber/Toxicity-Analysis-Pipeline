import os
from ray import init
import ray
from ray.data import read_parquet_bulk
from pathlib import Path

from src.config.config_loader import load_config
from src.utils.minhash_utils import LSHBuilder, MergeLSHActor

os.environ["RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S"] = "3600"

init(ignore_reinit_error=True)

def main():
    config = load_config()
    input_dir = config["dir"]["input_sample_dir"]
    output_dir = config["dir"]["lsh_dir"]
    output_file_name = "lsh.pkl"
    file_paths = [str(file) for file in Path(input_dir).glob("*.parquet")]

    merge_lsh = MergeLSHActor.remote()

    (
        read_parquet_bulk(
            paths=file_paths,
            concurrency=100,
            columns=["_id", "minhash"],
            ray_remote_args=dict(
                num_cpus=0.01,
                memory=2 * 1024**3,
                max_retries=10, 
                retry_exceptions=True,
            )
        )
        .map_batches(
            fn=LSHBuilder(),
            concurrency=100,
            num_cpus=0.01,
            batch_format="pandas",
            memory=16 * 1024**3,
            max_retries=10, 
            retry_exceptions=True,
        ) 
        .map_batches(
            lambda batch: ray.get(merge_lsh.merge.remote(batch)),
            concurrency=100,
            num_cpus=0.01,
            num_gpus=0,
            batch_format="pandas",
            memory=16 * 1024**3,
            max_retries=10, 
            retry_exceptions=True,       
        )
        .materialize()
    )

    ray.get(merge_lsh.store_lsh.remote(output_path=output_dir+output_file_name))

if __name__ == "__main__":
    main()
