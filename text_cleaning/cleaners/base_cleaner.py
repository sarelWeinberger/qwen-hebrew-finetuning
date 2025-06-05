import pandas as pd
from typing import Dict, Any
from ..logger import logger

class BaseCleaner:
    def __init__(self):
        self.stats = {
            'total_rows_processed': 0,
            'rows_modified': 0,
            'characters_removed': 0,
            'characters_added': 0,
            'patterns_matched': {},
            'execution_time': 0
        }
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input DataFrame.
        
        Args:
            df: Input DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cleaning statistics.
        
        Returns:
            Dictionary containing cleaning statistics
        """
        return self.stats
    
    def log_stats(self):
        """Log the cleaning statistics."""
        logger.info(f"Cleaning statistics for {self.__class__.__name__}:")
        logger.info(f"Total rows processed: {self.stats['total_rows_processed']}")
        logger.info(f"Rows modified: {self.stats['rows_modified']}")
        logger.info(f"Characters removed: {self.stats['characters_removed']}")
        logger.info(f"Characters added: {self.stats['characters_added']}")
        
        if self.stats['patterns_matched']:
            logger.info("Pattern matching statistics:")
            for pattern, count in self.stats['patterns_matched'].items():
                logger.info(f"  - Pattern '{pattern}': {count} matches")
        
        logger.info(f"Execution time: {self.stats['execution_time']:.2f} seconds")