import argparse
from pathlib import Path
import os

import ray
from ray.data import read_datasource, from_pandas
from ray.util.dask import enable_dask_on_ray, disable_dask_on_ray

from src.toxic_bert.toxicity_classifier_detoxify_original import ToxicityClassifierDetoxifyOriginal
from src.toxic_bert.toxicity_classifier_detoxify_unbiased import ToxicityClassifierDetoxifyUnbiased
from src.toxic_bert.toxicity_classifier_google import ToxicityClassifierGoogle
from src.toxic_bert.language_detection import LanguageDetector
from src.toxic_bert.similarity_grouping import GroupBuilder
from src.toxic_bert.config_loader import load_config
from src.toxic_bert.elasticsearch_utils import get_es_source
from src.toxic_bert.text_processing import extract_text
from src.toxic_bert.dataset import get_hash, write_data, merge_data

os.environ["RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S"] = "3600"

ray.init()

def main():
    parser = argparse.ArgumentParser(description="Run toxicity classification.")

    parser.add_argument(
        "--model",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Choose the model: 0 for Detoxify original, 1 for Detoxify unbiased, 2 for Google Perspective.",
    )
    args = parser.parse_args()

    if args.model == 0 :
        classifier = ToxicityClassifierDetoxifyOriginal()
        concurrency_classifier = 50
        cpu_classifier = 0.25
    elif args.model == 1:
        classifier = ToxicityClassifierDetoxifyUnbiased()
        concurrency_classifier = 50
        cpu_classifier = 0.25
    elif args.model == 2:
        classifier = ToxicityClassifierGoogle()
        concurrency_classifier = 50
        cpu_classifier = 0.2
    else:
        raise ValueError("Invalid model choice.")
    
    config_path = Path("data/config/config.yaml")
    config = load_config(config_path)
    output_dir = config["toxicity_analysis"]["output_dir"]
    es_source = get_es_source(config)

    ds = ( 
        read_datasource(
            datasource=es_source,
            concurrency=100,
            override_num_blocks=500,
            ray_remote_args=dict(
                num_cpus=0.01,
            ),
        )
        .map_batches(
            fn=extract_text,
            concurrency=500,
            num_cpus=0.01,
            batch_size=2048,
            batch_format="pandas",
            memory=5 * 1024**3, # 0.05 GB per task
        )
        .map_batches(
            fn=get_hash,
            concurrency=500,
            num_cpus=0.05,
            batch_size=1024,
            batch_format="pandas",
            memory=2 * 1024**3, # 0.1 GB per task
        ) 
        .map_batches(
            GroupBuilder(),
            concurrency=20,
            num_cpus=2,
            batch_size=1_000_000,
            batch_format="pandas",
            memory=10 * 1024**3, # 20 GB per task
        ) 
    )

    df = ds.to_pandas()
    df_small = df.drop(columns=["_id", "crawled_from_instance", "is_local", "created_at", "sensitive", "spoiler_text", "uri", "instance", "hash"])
    df_small = df_small.drop_duplicates(subset=["group"]).reset_index(drop=True)
    ds_small = from_pandas(df_small)

    ds_toxic = (
        ds_small
        .repartition(num_blocks=500)
        .map_batches(
            LanguageDetector(),
            concurrency=150,
            num_cpus=0.1,
            num_gpus=0,
            batch_size=1024,
            batch_format="pandas",
            memory=1 * 1024**3, # 0.1 GB per task        
        )
        .filter(
            lambda batch: batch['language'] == '__label__eng_Latn',
            concurrency=500,
            num_cpus=0.01,
            memory=5 * 1024**3, # 0.05 GB per task
        )
        .map_batches(
            classifier,
            concurrency=concurrency_classifier,
            num_cpus=cpu_classifier,
            num_gpus=0,
            batch_size=1024,
            batch_format="pandas",
            memory=4 * 1024**3, # 1 GB per task
        )
   )
    
    df_toxic = ds_toxic.to_pandas()
    df_toxic = df_toxic.drop(columns=["plaintext"])

    enable_dask_on_ray()
    merged_df = (
        merge_data(df_toxic, df, on="group")
        .dropna(subset=["language"])
        .drop(columns=["group", "language"])
    )
    disable_dask_on_ray()

    write_data(merged_df, output_dir)

if __name__ == "__main__":
    main()
