from pathlib import Path
from src.toxic_bert.config_loader import load_config
from src.toxic_bert.elasticsearch_utils import get_es_source
from src.toxic_bert.toxicity_analysis import ToxicityClassifier
from src.toxic_bert.text_processing import extract_text
from src.toxic_bert.language_detection import LanguageDetector

from ray import init
from ray.data import read_datasource, DataContext

init()
DataContext.get_current().enable_operator_progress_bars = True
RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=1

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
                "_source.content": "content",
                "_source.crawled_from_instance": "crawled_from_instance",
                "_source.instance": "instance",
                "_source.is_local": "is_local",
            }, 
            num_cpus=0.01,
        )
        # Map batches to extract text.
        .map_batches(
            fn=extract_text,
            num_cpus=0.01,
            batch_format="pandas",
        )
        # Map batches to add language tag.
        .map_batches(
            LanguageDetector(),
            concurrency=100,
            num_cpus=0.25,
            num_gpus=0,
            batch_format="pandas",
        )
        # Filter for English text.
        .filter(
            lambda batch: batch['language'] == '__label__eng_Latn'
        )
        # Classify toxiticity in batches.
        .map_batches(
            ToxicityClassifier(),
            concurrency=10,
            num_cpus=0.25,
            num_gpus=0,
            batch_size=32,
            batch_format="pandas",
        )
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
