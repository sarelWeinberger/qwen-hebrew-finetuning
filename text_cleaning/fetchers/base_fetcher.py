import pandas as pd
from typing import List, Dict, Any
import time
from logger import logger

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
        logger.info(f"Initialized {self.__class__.__name__} for source: {source_name}")

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
        logger.info(f"Fetching statistics for {self.__class__.__name__} - Source: {self.source_name}")
        logger.info(f"Total files processed: {self.stats['total_files_processed']}")
        logger.info(f"Total rows fetched: {self.stats['total_rows_fetched']}")
        logger.info(f"Total bytes read: {self.stats['total_bytes_read']}")
        logger.info(f"Total execution time: {self.stats['execution_time']:.2f} seconds")
        
        if self.stats['errors']:
            logger.warning(f"Encountered {len(self.stats['errors'])} errors:")
            for error in self.stats['errors']:
                logger.warning(f"  - {error}")
        
        if self.stats['file_stats']:
            logger.info("File-specific statistics:")
            for file_path, stats in self.stats['file_stats'].items():
                logger.info(f"  - {file_path}:")
                logger.info(f"    * Rows: {stats.get('rows', 0)}")
                logger.info(f"    * Size: {stats.get('size_bytes', 0)} bytes")
                logger.info(f"    * Processing time: {stats.get('processing_time', 0):.2f} seconds")