from .base_cleaner import BaseCleaner
import pandas as pd
import time
from utils.logger import logger


class CompositeCleaner(BaseCleaner):
    def __init__(self, cleaners: list[BaseCleaner]):
        """
        Initialize a composite cleaner that applies multiple cleaners in sequence.
        
        Args:
            cleaners: List of cleaner instances to apply in sequence
        """
        super().__init__()
        self.cleaners = cleaners
        logger.info(f"Initialized CompositeCleaner")

    def _clean_implementation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all cleaners in sequence to the input DataFrame.
        
        Args:
            df: Input DataFrame with 'text' column
            
        Returns:
            DataFrame with cleaned text and word count
        """
        current_df = df.copy()
        
        for i, cleaner in enumerate(self.cleaners, 1):
            cleaner_start_time = time.time()
            
            # Get initial stats
            initial_length = current_df['text'].str.len().sum()
            
            current_df = cleaner.clean(current_df)
            
            # Calculate changes
            final_length = current_df['text'].str.len().sum()
            chars_removed = max(0, initial_length - final_length)
            chars_added = max(0, final_length - initial_length)
            
            # Update composite stats
            self.stats['characters_removed'] += chars_removed
            self.stats['characters_added'] += chars_added
        
        return current_df