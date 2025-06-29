import pandas as pd
from typing import Dict, Any, Callable, List

class QualityCleaner:
    def __init__(self, metrics: Dict[str, Callable] = None):
        """
        Initialize the quality cleaner with configurable metrics.
        
        Args:
            metrics (Dict[str, Callable]): Dictionary mapping metric names to their calculation functions
        """
        self.metrics = metrics or {
            "single_char_percentage": self.calculate_single_char_percentage
        }


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

    def clean(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Add quality metrics to the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of the column containing text to analyze
            
        Returns:
            pd.DataFrame: Dataframe with added quality metrics
        """
        # Create a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Calculate and add each quality metric
        for metric_name, calculation_func in self.metrics.items():
            df[metric_name] = df[text_column].apply(calculation_func)
        
        return df
