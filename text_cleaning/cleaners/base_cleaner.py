import pandas as pd
from typing import Dict, Any, Optional
from utils.logger import logger
import time

class BaseCleaner:
    def __init__(self):
        """
        Initialize the base cleaner.
        
        """
        self.stats = {
            'total_rows_processed': 0,
            'rows_modified': 0,
            'characters_removed': 0,
            'characters_added': 0,
            'patterns_matched': {},
            'execution_time': 0
        }
        
    
    
    def clean(self, df: pd.DataFrame, file_name: str = "unknown") -> pd.DataFrame:
        """
        Clean the input DataFrame.
        
        Args:
            df: Input DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        start_time = time.time()
        
        
        # Perform the actual cleaning (to be implemented by subclasses)
        cleaned_df = self._clean_implementation(df)
        
        # Calculate statistics
        self.stats['total_rows_processed'] = len(df)
        self.stats['execution_time'] = time.time() - start_time
        
        # Calculate character changes
        if 'text' in df.columns and 'text' in cleaned_df.columns:
            initial_length = df['text'].str.len().sum()
            final_length = cleaned_df['text'].str.len().sum()
            self.stats['characters_removed'] = max(0, initial_length - final_length)
            self.stats['characters_added'] = max(0, final_length - initial_length)

        return cleaned_df
    
    def _clean_implementation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implementation of the cleaning logic. Must be overridden by subclasses.
        
        Args:
            df: Input DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        raise NotImplementedError("Subclasses must implement _clean_implementation")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cleaning statistics.
        
        Returns:
            Dictionary containing cleaning statistics
        """
        return self.stats
    
    def log_stats(self):
        """Log the cleaning statistics."""
        logger.info(f"Done {self.__class__.__name__}")
    