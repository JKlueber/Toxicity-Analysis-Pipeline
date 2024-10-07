from resiliparse.extract.html2text import extract_plain_text
import fasttext
from huggingface_hub import hf_hub_download
from pathlib import Path

from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))

def load_language_detector():
    model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    return fasttext.load_model(str(model_path))

def extract_text_from_hit(hit):
    text = extract_plain_text(
        hit['_source']['content'],
        main_content=True,
        alt_texts=False,
        preserve_formatting=False
    )
    
    filtered_words = [
        word for word in text.split() if word.lower() not in stopwords
    ]
    
    filtered_text = ' '.join(filtered_words)
    
    return filtered_text

def detect_language(text, lang_detector):
    return lang_detector.predict(text)[0][0]
