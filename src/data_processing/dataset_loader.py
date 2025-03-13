from pathlib import Path
from ray.data import Dataset

from ray.data import read_parquet

def load_dataset(data_path: str, pattern: str, concurrency: int = 100, num_blocks: int = 500) -> Dataset:
    file_paths = [str(file) for file in Path(data_path).glob(pattern)]
    
    ds = read_parquet(
        paths=file_paths,
        concurrency=concurrency,
        override_num_blocks=num_blocks,
        ray_remote_args=dict(num_cpus=0.01,
                             memory=2 * 1024**3)
    )
    return ds