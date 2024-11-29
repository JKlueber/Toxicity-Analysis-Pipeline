from pathlib import Path

from ray import init
from ray.data import read_datasource

from src.toxic_bert.config_loader import load_config
from src.toxic_bert.elasticsearch_utils import get_es_source
from src.toxic_bert.toxicity_analysis import ToxicityClassifier
from src.toxic_bert.text_processing import extract_text
from src.toxic_bert.language_detection import LanguageDetector

init()

def main():
    config_path = Path("data/config/config.yaml")
    config = load_config(config_path)

    # batch_size = config['toxicity_analysis']['batch_size']

    es_source = get_es_source(config)

    (
        read_datasource(
            datasource=es_source,
            concurrency=100,
            override_num_blocks=1000,
            ray_remote_args=dict(
                num_cpus=0.01,
            ),
        )
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
            batch_format="pandas",
        )
        # Filter for English text.
        .filter(
            lambda batch: batch['language'] == '__label__eng_Latn',
            concurrency=500,
            num_cpus=0.01,
        )
        # Classify toxiticity in batches.
        .map_batches(
            ToxicityClassifier(),
            concurrency=100,
            num_cpus=0.25,
            num_gpus=0,
            # batch_size=32,
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
