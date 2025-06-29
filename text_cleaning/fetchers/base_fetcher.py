import pandas as pd
from typing import List, Dict, Any
import time
import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import logger

class BaseFetcher:
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.stats = {
            'total_files_processed': 0,
            'total_rows_fetched': 0,
            'total_bytes_read': 0,
            'execution_time': 0,
            'errors': [],
            'file_stats': {}
        }
        logger.info(f"Initialized {self.__class__.__name__}")

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
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get fetching statistics.
        
        Returns:
            Dictionary containing fetching statistics
        """
        return self.stats
    
    def log_stats(self):
        """Log the fetching statistics."""
        logger.info(f"Done {self.__class__.__name__}")