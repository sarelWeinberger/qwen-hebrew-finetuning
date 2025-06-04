import pandas as pd
class BaseFetcher:
    def __init__(self, source_name: str):
        self.source_name = source_name

    def fetch_raw_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def save_cleaned_data(self, df: pd.DataFrame, source_name: str):
        raise NotImplementedError