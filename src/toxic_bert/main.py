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
            concurrency=10,
            override_num_blocks=50#0,
        )
        # .rename_columns({
        #     "_source.content": "content",
        #     "_source.crawled_from_instance": "crawled_from_instance",
        #     "_source.instance": "instance",
        #     "_source.is_local": "is_local",
        # })
        # .map_batches(
        #     fn=extract_text,
        #     batch_format="pandas",
        # )
        # .map_batches(
        #     LanguageDetector(),
        #     concurrency=10,
        #     num_cpus=0.25,
        #     num_gpus=0,
        #     # batch_size=batch_size,
        #     batch_format="pandas",
        # )
        # .filter(
        #     lambda batch: batch['language'] == '__label__eng_Latn'
        # )
        # .map_batches(
        #     ToxicityClassifier(),
        #     concurrency=10,
        #     num_cpus=0.25,
        #     num_gpus=0,
        #     # batch_size=batch_size,
        #     batch_format="pandas",
        # )
        .write_json("local:///mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-klueber/toxicity")
    )

if __name__ == "__main__":
    main()
