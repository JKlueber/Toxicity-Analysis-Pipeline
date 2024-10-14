from pathlib import Path
from config_loader import load_config
from elasticsearch_utils import connect_to_elastic, prepare_search_query
from text_processing import load_language_detector
from toxicity_analysis import load_toxicity_model, measure_toxicity
from elasticsearch_dsl.query import Term
import pandas as pd
import os

import ray
from ray import init, remote, get
from ray.data.datasource import FilenameProvider

ray.init(logging_level="DEBUG")

@remote
def process_toxicity(config_path):
    config = load_config(config_path)

    es = connect_to_elastic(config)
    base_search = prepare_search_query(es, config)

    language = config['elasticsearch']['language']
    filtered_search = base_search.filter(Term(language=language))

    lang_detector = load_language_detector()
    toxic_bert = load_toxicity_model()

    batch_size = config['toxicity_analysis']['batch_size']
    index = config['elasticsearch']['index']
    num_of_res = 10

    time, cutted, false_lang, toxicitys = measure_toxicity(filtered_search, es, index, lang_detector, toxic_bert, batch_size, num_of_res)

    return {
        "time": time,
        "batch_size": batch_size,
        "num_of_results": num_of_res,
        "cutted": cutted,
        "false_language": false_lang,
        "toxicitys": toxicitys
    }

def main():
    config_path = Path("data/config/config.yaml")
    
    future = process_toxicity.remote(config_path)

    result = get(future)
    toxicitys = result['toxicitys']

    df = pd.DataFrame(toxicitys)
    ds = ray.data.from_pandas([df])

    ds.write_json(
    "local:///mnt/ceph/storage/data-tmp/2024/po87xox/toxicity",
    ray_remote_args={"resources": {"num_cpus": 1}, "scheduling_strategy": "DEFAULT"}
    )

    print(f"Time taken: {result['time']} seconds")
    print(f"Batch size: {result['batch_size']}")
    print(f"Number of results: {result['num_of_results']}")
    print(f"Number of cutted texts: {result['cutted']}")
    print(f"Number of texts with false language: {result['false_language']}")

if __name__ == "__main__":
    main()

