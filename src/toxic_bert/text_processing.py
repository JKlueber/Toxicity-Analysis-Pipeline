from resiliparse.extract.html2text import extract_plain_text
from pandas import DataFrame

import gc

def extract_text(batch: DataFrame) -> DataFrame:
    # Extract plaintext from HTML content
    batch["plaintext"] = batch['content'].apply(
        lambda content: extract_plain_text(content, main_content=True, alt_texts=False, preserve_formatting=False)
    )

    # Filter out empty or whitespace-only plaintexts
    batch = batch[batch['plaintext'].str.strip().astype(bool)]

    # Drop the original content column
    batch = batch.drop(columns=['content'])

    gc.collect()

    return batch
