from functools import cached_property

from torch import device
from torch.cuda import is_available as cuda_is_available
from pandas import DataFrame
from transformers import pipeline, Pipeline

import gc


class ToxicityClassifierDetoxifyOriginal:

    def __init__(self):
        self._pipeline = self._load_pipeline

    @cached_property
    def _load_pipeline(self) -> Pipeline:
        return pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            tokenizer="bert-base-uncased",
            top_k=None,  # all labels
            truncation=True,
            padding=True,
            device=device("cuda" if cuda_is_available() else "cpu"),
        )

    def __call__(self, batch: DataFrame) -> DataFrame:
        # Reset call count of classifier pipeline.
        self._pipeline.call_count = 0

        # Predict full batch in one call to the model (pipeline).
        predictions = self._pipeline(list(batch["plaintext"]))

        # Extract the individual label scores per prediction.
        scores_per_label = [
            {item["label"]: item["score"] for item in prediction}
            for prediction in predictions
        ]

        # Store each label's scores into separate columns.
        labels_to_columns = {
            "toxic": "toxicity",
            "severe_toxic": "severe_toxicity",
            "obscene": "obscenity",
            "threat": "threat",
            "insult": "insult",
            "identity_hate": "identity_attack",
        }
        for label, column in labels_to_columns.items():
            batch[column] = [
                scores.get(label, 0)
                for scores in scores_per_label
            ]

        gc.collect()    

        return batch
