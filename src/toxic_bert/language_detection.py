from functools import cached_property

from fasttext import load_model
from fasttext.FastText import _FastText
from huggingface_hub import hf_hub_download
from pandas import DataFrame

import gc


class LanguageDetector:

    @cached_property
    def _lang_detector(self) -> _FastText:
        model_path = hf_hub_download(
            repo_id="facebook/fasttext-language-identification",
            cache_dir="/tmp/fasttext",
            filename="model.bin",
        )
        return load_model(model_path)

    def __call__(self, batch: DataFrame) -> DataFrame:
        # Predict full batch in one call to the model (fastText).
        multi_labels, _ = self._lang_detector.predict(
            [text.replace("\n", " ") for text in batch["plaintext"]]
        )
        batch["language"] = [labels[0] for labels in multi_labels]

        gc.collect()

        return batch
