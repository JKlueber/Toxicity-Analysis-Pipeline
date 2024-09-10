from resiliparse.extract.html2text import extract_plain_text
import fasttext
from pathlib import Path

def load_language_detector(model_path: Path):
    return fasttext.load_model(str(model_path))

def extract_text_from_hit(hit):
    return extract_plain_text(
        hit['_source']['content'],
        main_content=True,
        alt_texts=False,
        preserve_formatting=False
    )

def detect_language(text, lang_detector):
    return lang_detector.predict(text)[0][0]
