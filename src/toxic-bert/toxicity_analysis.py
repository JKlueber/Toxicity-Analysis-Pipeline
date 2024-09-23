import torch
from transformers import pipeline
from elasticsearch_utils import execute_scan
from text_processing import detect_language, extract_text_from_hit
import json
import os

class Toxicity:
    def __init__(self, toxicity=0, severe_toxicity=0, obscene=0, threat=0, insult=0, identity_attack=0):
        self.toxicity = toxicity
        self.severe_toxicity = severe_toxicity
        self.obscene = obscene
        self.threat = threat
        self.insult = insult
        self.identity_attack = identity_attack
    
    def __add__(self, other):
        return Toxicity(
            self.toxicity + other.toxicity,
            self.severe_toxicity + other.severe_toxicity,
            self.obscene + other.obscene,
            self.threat + other.threat,
            self.insult + other.insult,
            self.identity_attack + other.identity_attack
        )
    
    def __truediv__(self, other):
        return Toxicity(
            self.toxicity / other,
            self.severe_toxicity / other,
            self.obscene / other,
            self.threat / other,
            self.insult / other,
            self.identity_attack / other
        )
    
    def __str__(self):
        return f"Toxicity: {self.toxicity}, Severe Toxicity: {self.severe_toxicity}, Obscene: {self.obscene}, Threat: {self.threat}, Insult: {self.insult}, Identity Attack: {self.identity_attack}"

    @classmethod
    def from_prediction(cls, prediction):
        scores = {item['label']: item['score'] for item in prediction[0]}
        return cls(
            toxicity=scores.get('toxic', 0),
            severe_toxicity=scores.get('severe_toxic', 0),
            obscene=scores.get('obscene', 0),
            threat=scores.get('threat', 0),
            insult=scores.get('insult', 0),
            identity_attack=scores.get('identity_hate', 0)
        )
    
    def to_dict(self):
        return {
            'toxicity': self.toxicity,
            'severe_toxicity': self.severe_toxicity,
            'obscene': self.obscene,
            'threat': self.threat,
            'insult': self.insult,
            'identity_attack': self.identity_attack
        }

def load_toxicity_model():
    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        'text-classification', 
        model='unitary/toxic-bert', 
        tokenizer='bert-base-uncased', 
        top_k=None, 
        device=device 
    )
    

def measure_toxicity(filtered_search, es, index, lang_detector, toxic_bert, batch_size=128, output_path='data/output/toxicity_results.json'):
    toxicitys = []
    batch = []

    it = execute_scan(filtered_search, es, index, size=10)
    count = 0
    for hit in it:
        count += 1
        plaintext = extract_text_from_hit(hit)
        lang = detect_language(plaintext, lang_detector)
        hit_id = hit.get('_id')
        crawled_from_instance = hit['_source'].get('crawled_from_instance')
        instance = hit['_source'].get('instance')
        is_local = hit['_source'].get('is_local')

        if lang == '__label__eng_Latn':
            batch.append({
                'text': plaintext,
                'id': hit_id,
                'crawled_from_instance': crawled_from_instance,
                'instance': instance,
                'is_local': is_local
            })
            if len(batch) == batch_size:
                predictions = toxic_bert([item['text'] for item in batch]) 
                print(predictions) 
                for item, prediction in zip(batch, predictions):
                    toxicity = Toxicity.from_prediction([prediction])
                    toxicitys.append({
                        'id': item['id'],
                        'crawled_from_instance': item['crawled_from_instance'],
                        'instance': item['instance'],
                        'is_local': item['is_local'],
                        'toxicity': toxicity.to_dict()
                    })
                batch = []
        if count > 5:
            break

    if batch:
        predictions = toxic_bert([item['text'] for item in batch])
        for item, prediction in zip(batch, predictions):
            toxicity = Toxicity.from_prediction([prediction])
            toxicitys.append({
                'id': item['id'],
                'crawled_from_instance': item['crawled_from_instance'],
                'instance': item['instance'],
                'is_local': item['is_local'],
                'toxicity': toxicity.to_dict()
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as json_file:
        json.dump(toxicitys, json_file, indent=4) 

    return toxicitys
