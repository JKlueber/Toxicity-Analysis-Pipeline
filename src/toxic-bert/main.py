import os
from pathlib import Path
from config_loader import load_config
from elasticsearch_utils import connect_to_elastic, prepare_search_query
from text_processing import load_language_detector
from toxicity_analysis import load_toxicity_model, measure_toxicity

from elasticsearch_dsl.query import Term
import csv


def main():
    config_path = Path("data/config/config.yaml")
    config = load_config(config_path)

    es = connect_to_elastic(config)
    base_search = prepare_search_query(es, config)
    
    language = config['elasticsearch']['language']

    filtered_search = base_search.filter(Term(language=language))

    lang_detector = load_language_detector()
    toxic_bert = load_toxicity_model()

    batch_size = config['toxicity_analysis']['batch_size']
    index = config['elasticsearch']['index']

    num_of_res = 100000

    time, cutted, false_lang = measure_toxicity(filtered_search, es, index, lang_detector, toxic_bert, batch_size, num_of_res)
    print(f"Time taken: {time:.2f} seconds")
    print(f"Batch size: {batch_size}")
    print(f"Number of results: {num_of_res}")
    print(f"Number of cutted texts: {cutted}")
    print(f"Number of texts with false language: {false_lang}")


if __name__ == "__main__":
    main()
