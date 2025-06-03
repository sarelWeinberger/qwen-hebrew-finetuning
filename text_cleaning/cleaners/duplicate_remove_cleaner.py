from .base_cleaner import BaseCleaner
import pandas as pd

class DuplicateRemoverCleaner(BaseCleaner):
    def __init__(self, text_column: str = "original_text"):
        self.text_column = text_column

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.text_column not in df.columns:
            raise ValueError(f"Column '{self.text_column}' not found in dataframe")
        
        return df.drop_duplicates(subset=[self.text_column]).reset_index(drop=True)

    def clean(self, raw_text: str) -> str:
        return raw_text