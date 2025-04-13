from pandas import DataFrame
from googleapiclient import discovery
from dotenv import load_dotenv
import os

import time
import random
from googleapiclient.errors import HttpError

import gc

load_dotenv()
API_KEY = os.getenv('PERSPECTIVE_API_KEY')

class ToxicityClassifierPerspectiveAPI:

    # Initialize the Perspective API client
    def __init__(self):
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        )

    def _analyze(self, text: str) -> dict:

        max_bytes = 20400
        if len(text.encode('utf-8')) > max_bytes:
            while len(text.encode('utf-8')) > max_bytes:
                text = text[:-10]

        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {
                'TOXICITY': {},
                'SEVERE_TOXICITY': {},
                'IDENTITY_ATTACK': {},
                'INSULT': {},
                'PROFANITY': {},
                'THREAT': {},
                'SEXUALLY_EXPLICIT': {},
            },
            'languages': ['en']
        }

        retries = 10
        delay = 1  # Start delay in seconds
        for i in range(retries):
            try:
                response = self.client.comments().analyze(body=analyze_request).execute()

                # Extract scores for all requested attributes
                return {
                    attr: response['attributeScores'][attr]['summaryScore']['value']
                    for attr in analyze_request['requestedAttributes']
                }

            except HttpError as e:
                if e.resp.status == 429:  # Quota exceeded
                    wait_time = delay * (2 ** i) + random.uniform(0, 1)
                    print(f"Quota exceeded, retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    raise e  # If it's not a rate limit error, raise the exception
        raise Exception("Max retries reached, still receiving quota exceeded errors.")

    def __call__(self, batch: DataFrame) -> DataFrame:
        # Predict full batch in sequence of calls to the model (API).
        texts = list(batch["plaintext"])
        predictions = []
        for text in texts:
            predictions.append(self._analyze(text))
            time.sleep(1)

        # Store each label's scores into separate columns.
        labels_to_columns = {
            "TOXICITY": "toxicity",
            "SEVERE_TOXICITY": "severe_toxicity",
            "PROFANITY": "obscenity",
            "THREAT": "threat",
            "INSULT": "insult",
            "IDENTITY_ATTACK": "identity_attack",
            "SEXUALLY_EXPLICIT": "sexually_explicit",
        }
        for label, column in labels_to_columns.items():
            batch[column] = [
                prediction.get(label, 0) 
                for prediction in predictions
            ]
    
        gc.collect()

        return batch