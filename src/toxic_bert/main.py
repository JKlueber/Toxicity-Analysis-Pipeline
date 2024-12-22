import argparse

from pathlib import Path

from ray import init
from ray.data import read_json
from ray.data import read_datasource

from src.toxic_bert.config_loader import load_config
from src.toxic_bert.elasticsearch_utils import get_es_source
from src.toxic_bert.toxicity_classifier_detoxify_original import ToxicityClassifierDetoxifyOriginal
from src.toxic_bert.toxicity_classifier_detoxify_unbiased import ToxicityClassifierDetoxifyUnbiased
from src.toxic_bert.toxicity_classifier_google import ToxicityClassifierGoogle
from src.toxic_bert.text_processing import extract_text
from src.toxic_bert.language_detection import LanguageDetector

init()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run toxicity classification.")

    parser.add_argument(
        "--model",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Choose the model: 0 for Detoxify original, 1 for Detoxify unbiased, 2 for Google Perspective.",
    )

    args = parser.parse_args()

    config_path = Path("data/config/config.yaml")
    config = load_config(config_path)

    # batch_size = config['toxicity_analysis']['batch_size']

    es_source = get_es_source(config)
    if args.model == 0 :
        classifier = ToxicityClassifierDetoxifyOriginal()
        concurrency_classifier = 50
    elif args.model == 1:
        classifier = ToxicityClassifierDetoxifyUnbiased()
        concurrency_classifier = 50
    elif args.model == 2:
        classifier = ToxicityClassifierGoogle()
        concurrency_classifier = 1
    else:
        raise ValueError("Invalid model choice.")


    (
        read_datasource(
            datasource=es_source,
            concurrency=100,
            override_num_blocks=1000,
            ray_remote_args=dict(
                num_cpus=0.01,
            ),
        )
        # read_json("/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-klueber/toxicity")
        # Rename the columns.
        .rename_columns(
            names={
                "_id": "id",
            },
            concurrency=500,
            num_cpus=0.01,
        )
        # Map batches to extract text.
        .map_batches(
            fn=extract_text,
            concurrency=500,
            num_cpus=0.05,
            batch_format="pandas",
        )
        # Map batches to add language tag.
        .map_batches(
            LanguageDetector(),
            concurrency=500,
            num_cpus=0.25,
            num_gpus=0,
            batch_size=256,
            # memory = 10 * 1024**3,
            batch_format="pandas",
        )
        # Filter for English text.
        .filter(
            lambda batch: batch['language'] == '__label__eng_Latn',
            concurrency=500,
            num_cpus=0.01,
            # memory = 5 * 1024**3,
        )
        # Classify toxiticity in batches.
        .map_batches(
            classifier,
            concurrency=concurrency_classifier,
            num_cpus=0.75,
            num_gpus=0,
            memory = 20 * 1024**3,
            batch_format="pandas",
        )
        # .limit(1000)
        .write_json(
            path="/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-klueber/toxicity",
            concurrency=10,
            ray_remote_args=dict(
                num_cpus=0.1,
            ),
        )
    )

if __name__ == "__main__":
    main()
