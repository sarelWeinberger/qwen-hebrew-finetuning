import pandas as pd
from typing import List

class BaseFetcher:
    def __init__(self, source_name: str):
        self.source_name = source_name

    def get_files_to_process(self) -> List[str]:
        """
        Returns a list of file paths that need to be processed.
        Each fetcher implementation should define how to get this list.
        """
        raise NotImplementedError

    def fetch_single_file(self, file_path: str) -> pd.DataFrame:
        """
        Fetches and returns data from a single file.
        """
        raise NotImplementedError

    def save_cleaned_data(self, df: pd.DataFrame, source_name: str, original_file_path: str):
        """
        Saves cleaned data, taking into account the original file path.
        """
        raise NotImplementedError