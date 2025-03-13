import argparse
from pathlib import Path
import os

import ray

from src.toxicity_predicting.toxicity_classifier_detoxify_original import ToxicityClassifierDetoxifyOriginal
from src.toxicity_predicting.toxicity_classifier_detoxify_unbiased import ToxicityClassifierDetoxifyUnbiased
from src.toxicity_predicting.toxicity_classifier_perspective_api import ToxicityClassifierPerspectiveAPI
from src.data_processing.language_detector import LanguageDetector
from src.config.config_loader import load_config
from src.data_processing.dataset_loader import load_dataset

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


    config_path = Path("src/config/config.yaml")
    config = load_config(config_path)

    input_dir = config["deduplication"]["output_dir"]
    output_dir = config["toxicity_analysis"]["output_dir"]

    ds = load_dataset(input_dir, "*.parquet")

    (
        ds
        #.repartition(num_blocks=ds.count() // 1000)
        .map_batches(
            LanguageDetector(),
            concurrency=500,
            num_cpus=0.1,
            num_gpus=0,
            batch_format="pandas",
            memory=3 * 1024**3,       
        )
        .filter(
            lambda batch: batch['language'] == '__label__eng_Latn',
            concurrency=500,
            num_cpus=0.01,
            memory=2 * 1024**3,
        )
        .map_batches(
            classifier,
            concurrency=150,
            num_cpus=0.5,
            num_gpus=0,
            batch_format="pandas",
            memory=5 * 1024**3,
        )
        .write_parquet(
            path=output_dir,
            ray_remote_args=dict(
                num_cpus=0.01,
                memory = 2 * 1024**3,
            )
        )
    )

if __name__ == "__main__":
    main()