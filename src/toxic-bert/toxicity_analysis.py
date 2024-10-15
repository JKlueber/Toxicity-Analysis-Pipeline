from transformers import pipeline
from text_processing import detect_language, extract_text_from_hit
from text_processing import load_language_detector
import torch
from typing import Dict
import numpy as np

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
    # device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        'text-classification', 
        model='unitary/toxic-bert', 
        tokenizer='bert-base-uncased', 
        top_k=None, 
    )

class ToxicityClassifier:
    def __init__(self):
        self.classifier = None
        self.lang_detector = None

    def initialize(self):
        if self.classifier is None:
            self.classifier = load_toxicity_model()
        if self.lang_detector is None:
            self.lang_detector = load_language_detector()
    
    def __call__(self, batch: Dict[str, np.ndarray]):
        self.initialize()
        toxicity_results = []
        texts = []

        for item in batch:
            try:
                text = extract_text_from_hit(item)
                lang = detect_language(text, self.lang_detector)

                if lang == '__label__eng_Latn':
                    truncated_text = text[:512] if len(text) > 512 else text
                    texts.append(truncated_text)
            except Exception as e:
                print(f"Error processing item {item['_source'].get('_id', 'unknown')}: {e}")
                continue 

        if texts:
            try:
                predictions = self.classifier(texts)
                for item, prediction in zip(batch, predictions):
                    toxicity = Toxicity.from_prediction([prediction])
                    toxicity_results.append({
                        'id': item.get('_id'),
                        'crawled_from_instance': item['_source'].get('crawled_from_instance'),
                        'instance': item['_source'].get('instance'),
                        'is_local': item['_source'].get('is_local'),
                        'toxicity': toxicity.to_dict()
                    })
            except Exception as e:
                print(f"Error during classification: {e}")
                return []

        return toxicity_results

