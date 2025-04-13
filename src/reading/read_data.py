# +----------------+----------------+
# |    Month       |  Toots Count   |
# +----------------+----------------+
# | January        |   143,514,788  |
# | February       |   138,951,038  |
# | March          |   146,515,289  |
# | April          |   136,869,920  |
# | May            |   148,136,851  |
# | June           |   139,149,320  |
# | July           |   146,365,161  |
# | August         |   144,063,875  |
# | September      |   149,418,345  |
# | October        |   176,772,707  |
# | November       |   167,360,640  |
# | December       |   150,255,622  |
# +----------------+----------------+
# | total          | 1,787,373,556  |  from 1000 instances without reblogs and without media attachments
# +----------------+----------------+

import os
from ray import init
from ray.data import read_datasource
import pandas as pd

from src.utils.elasticsearch_utils import get_es_source
from src.config.config_loader import load_config

def sample_batch(batch: pd.DataFrame) -> pd.DataFrame:
    return batch.sample(frac=0.1, random_state=42)

os.environ["RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S"] = "3600"

init(runtime_env={"config": {"setup_timeout_seconds": 3600}})

def main():
    config = load_config()
    output_dir = config["dir"]["input_dir"]
    es_source = get_es_source(config)

    (
        read_datasource(
            datasource=es_source,
            concurrency=100,
            override_num_blocks=500,
            ray_remote_args=dict(
                num_cpus=0.01,
                memory=2 * 1024**3,
                max_retries=10, 
                retry_exceptions=True,
            ),
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
