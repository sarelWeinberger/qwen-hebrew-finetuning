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
        logger.info(f"Initialized CompositeCleaner with {len(cleaners)} cleaners")
        for i, cleaner in enumerate(cleaners, 1):
            logger.info(f"Cleaner {i}: {cleaner.__class__.__name__}")

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all cleaners in sequence to the input DataFrame.
        
        Args:
            df: Input DataFrame with 'text' column
            
        Returns:
            DataFrame with cleaned text and word count
        """
        start_time = time.time()
        current_df = df.copy()
        self.stats['total_rows_processed'] = len(df)
        
        logger.info(f"Starting cleaning pipeline on {len(df)} rows")
        
        for i, cleaner in enumerate(self.cleaners, 1):
            cleaner_start_time = time.time()
            logger.info(f"Applying cleaner {i}/{len(self.cleaners)}: {cleaner.__class__.__name__}")
            
            # Get initial stats
            initial_length = current_df['text'].str.len().sum()
            
            # Apply cleaner
            current_df = cleaner.clean(current_df)
            
            # Calculate changes
            final_length = current_df['text'].str.len().sum()
            chars_removed = max(0, initial_length - final_length)
            chars_added = max(0, final_length - initial_length)
            
            # Update composite stats
            self.stats['characters_removed'] += chars_removed
            self.stats['characters_added'] += chars_added
            
            # Log cleaner-specific stats
            cleaner_stats = cleaner.get_stats()
            logger.info(f"Cleaner {i} statistics:")
            logger.info(f"  - Rows modified: {cleaner_stats['rows_modified']}")
            logger.info(f"  - Characters removed: {chars_removed}")
            logger.info(f"  - Characters added: {chars_added}")
            logger.info(f"  - Execution time: {time.time() - cleaner_start_time:.2f} seconds")
            
            if cleaner_stats['patterns_matched']:
                logger.info("  - Pattern matches:")
                for pattern, count in cleaner_stats['patterns_matched'].items():
                    logger.info(f"    * {pattern}: {count} matches")
        
        self.stats['execution_time'] = time.time() - start_time
        
        # Log final statistics
        logger.info("Cleaning pipeline completed")
        logger.info(f"Total execution time: {self.stats['execution_time']:.2f} seconds")
        logger.info(f"Total characters removed: {self.stats['characters_removed']}")
        logger.info(f"Total characters added: {self.stats['characters_added']}")
        
        return current_df