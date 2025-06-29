import pandas as pd
from typing import Dict, Any, Callable, List
from .base_cleaner import BaseCleaner
from utils.logger import logger

class QualityCleaner(BaseCleaner):
    def __init__(self, metrics: Dict[str, Callable] = None, save_samples: bool = True, sample_percentage: float = 0.05):
        """
        Initialize the quality cleaner with configurable metrics.
        
        Args:
            metrics (Dict[str, Callable]): Dictionary mapping metric names to their calculation functions
            save_samples: Whether to save before/after samples
            sample_percentage: Percentage of data to sample
        """
        super().__init__(save_samples=save_samples, sample_percentage=sample_percentage)
        self.metrics = metrics or {
            "single_char_percentage": self.calculate_single_char_percentage
        }
        logger.info(f"Initialized QualityCleaner with {len(self.metrics)} metrics")

    def calculate_single_char_percentage(self, text: str) -> float:
        """
        Calculate the percentage of characters that appear exactly once in the text.
        For example: "a a b b cc" would return 4/5 = 0.8 (80%)
        as 'a' appears once, 'a' appears once, 'b' appears once, 'b' appears once (4 total)
        out of 5 total non-space characters.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            float: Percentage of single-occurrence characters (0-100)
        """
        if not text or not isinstance(text, str):
            return 0.0
            
        # Remove spaces and count all characters
        text_no_spaces = text.replace(" ", "")
        if not text_no_spaces:
            return 0.0
            
        # Count all characters
        char_counts = {}
        for char in text_no_spaces:
            char_counts[char] = char_counts.get(char, 0) + 1
            
        # Count total occurrences of characters that appear exactly once
        single_chars = sum(count for count in char_counts.values() if count == 1)
        total_chars = len(text_no_spaces)
        
        return (single_chars / total_chars) * 100

    def add_metric(self, name: str, calculation_func: Callable) -> None:
        """
        Add a new quality metric to the cleaner.
        
        Args:
            name (str): Name of the metric
            calculation_func (Callable): Function that calculates the metric
        """
        self.metrics[name] = calculation_func
        logger.info(f"Added quality metric: {name}")

    def _clean_implementation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add quality metrics to the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe with 'text' column
            
        Returns:
            pd.DataFrame: Dataframe with added quality metrics and updated 'n_words'
        """
        # Create a copy to avoid modifying the original dataframe
        result_df = df.copy()
        
        # Calculate and add each quality metric
        for metric_name, calculation_func in self.metrics.items():
            result_df[metric_name] = result_df['text'].apply(calculation_func)
        
        # Update word count
        result_df['n_words'] = result_df['text'].str.split().str.len()
        
        # Calculate rows modified (any row that has new metrics added)
        self.stats['rows_modified'] = len(result_df)  # All rows get metrics added
        
        logger.info(f"QualityCleaner processed {len(df)} rows")
        logger.info(f"Added {len(self.metrics)} quality metrics to all rows")
        
        return result_df
