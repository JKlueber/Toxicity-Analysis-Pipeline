import argparse
from pathlib import Path
import os

import ray
from ray.data import read_parquet_bulk

from src.toxicity_predicting.toxicity_classifier_detoxify_original import ToxicityClassifierDetoxifyOriginal
from src.toxicity_predicting.toxicity_classifier_detoxify_unbiased import ToxicityClassifierDetoxifyUnbiased
from src.toxicity_predicting.toxicity_classifier_perspective_api import ToxicityClassifierPerspectiveAPI
from src.data_processing.language_detector import LanguageDetector
from src.config.config_loader import load_config

os.environ["RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S"] = "3600"

ray.init()

def main():
    parser = argparse.ArgumentParser(description="Run toxicity classification.")

    parser.add_argument(
        "--model",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Choose the model: 0 for Detoxify original, 1 for Detoxify unbiased, 2 for Perspective API.",
    )
    args = parser.parse_args()

    if args.model == 0 :
        classifier = ToxicityClassifierDetoxifyOriginal()
    elif args.model == 1:
        classifier = ToxicityClassifierDetoxifyUnbiased()
    elif args.model == 2:
        classifier = ToxicityClassifierPerspectiveAPI()
    else:
        raise ValueError("Invalid model choice.")


    config = load_config()

    input_dir = config["dir"]["deduplication_dir"]
    output_dir = config["dir"]["toxicity_analysis_dir"]
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
            LanguageDetector(),
            concurrency=100,
            num_cpus=0.01,
            num_gpus=0,
            batch_format="pandas",
            memory=2 * 1024**3,
            max_retries=10, 
            retry_exceptions=True,       
        )
        .filter(
            lambda batch: batch['language'] == '__label__eng_Latn',
            concurrency=100,
            num_cpus=0.01,
            memory=2 * 1024**3,
            max_retries=10, 
            retry_exceptions=True,
        )
        .map_batches(
            classifier,
            concurrency=100,
            num_cpus=0.01,
            num_gpus=0,
            batch_format="pandas",
            memory=3 * 1024**3,
            max_retries=10, 
            retry_exceptions=True,
        )
        .write_parquet(
            path=output_dir,
            concurrency=100,
            ray_remote_args=dict(
                num_cpus=0.01,
                memory =2 * 1024**3,
                max_retries=10, 
                retry_exceptions=True,
            )
        )
    )

if __name__ == "__main__":
    main()