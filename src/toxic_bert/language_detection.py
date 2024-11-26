from pandas import DataFrame
import fasttext
from huggingface_hub import hf_hub_download

def load_language_detector():
    model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    return fasttext.load_model(str(model_path))

def detect_language(plaintext, lang_detector):
    return lang_detector.predict(plaintext)[0][0]

class LanguageDetector:
    def __init__(self):
        self.lang_detector = None

    def initialize(self):
        if self.lang_detector is None:
            self.lang_detector = load_language_detector()

    def __call__(self, batch: DataFrame) -> DataFrame:
        self.initialize()
        batch['language'] = batch['plaintext'].apply(
            lambda plaintext: detect_language(plaintext, lang_detector=self.lang_detector)
        )
        return batch
