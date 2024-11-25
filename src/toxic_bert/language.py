from pandas import DataFrame

def detect_language(batch: DataFrame) -> DataFrame:
    raise NotImplementedError()


def is_english(row) -> bool:
    raise NotImplementedError()
