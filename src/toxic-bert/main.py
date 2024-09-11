import os
from pathlib import Path
from config_loader import load_config
from elasticsearch_utils import connect_to_elastic, prepare_search_query
from text_processing import load_language_detector
from toxicity_analysis import load_toxicity_model, measure_toxicity

from elasticsearch_dsl.query import Term
import csv


def main():
    config_path = Path("../data/config/config.yaml")
    config = load_config(config_path)

    es = connect_to_elastic(config)
    base_search = prepare_search_query(es, config)
    
    language = config['elasticsearch']['language']

    filtered_search = base_search.filter(Term(language=language))

    lang_detector = load_language_detector()
    toxic_bert = load_toxicity_model().to('cuda')

    batch_size = config['toxicity_analysis']['batch_size']
    index = config['elasticsearch']['index']

    toxicitys = measure_toxicity(filtered_search, es, index, lang_detector, toxic_bert, batch_size=2)
    print(toxicitys)

    # output_path = Path(f"../data/toxicity_results_{os.getenv('SLURM_ARRAY_TASK_ID', '0')}.txt")
    # with output_path.open("w", newline='') as csvfile:
    #     fieldnames = ['instance', 'toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
    #     writer.writeheader()
    #     for result in results:
    #         writer.writerow(result)


if __name__ == "__main__":
    main()
