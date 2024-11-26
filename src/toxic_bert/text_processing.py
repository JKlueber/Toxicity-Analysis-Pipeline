from resiliparse.extract.html2text import extract_plain_text
from pandas import DataFrame

def extract_text(batch: DataFrame) -> DataFrame:
    batch["plaintext"] = batch['content'].apply(
        lambda content: extract_plain_text(content, main_content=True, alt_texts=False, preserve_formatting=False)
    )
    return batch
