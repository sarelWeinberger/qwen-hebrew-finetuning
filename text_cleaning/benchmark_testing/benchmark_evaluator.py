import pandas as pd
import numpy as np
from Levenshtein import distance
from typing import List, Tuple, Dict, Union
import json
from pathlib import Path


class BenchmarkEvaluator:
    def __init__(self, 
                 benchmark_file: str,
                 manual_clean_col: str = 'manual_clean',
                 cleaned_text_col: str = 'cleaned_text'):
        """
        Initialize the benchmark evaluator with a benchmark file.
        
        Args:
            benchmark_file (str): Path to the benchmark file (Excel) containing raw and cleaned text pairs
            manual_clean_col (str): Name of the column containing manually cleaned text
            cleaned_text_col (str): Name of the column containing LLM cleaned text
        """
        self.manual_clean_col = manual_clean_col
        self.cleaned_text_col = cleaned_text_col
        self.benchmark_data = {}
        self.results = {}
        
        # Read Excel file with multiple sheets
        excel_data = pd.read_excel(benchmark_file, sheet_name=None)
        for sheet_name, df in excel_data.items():
            if manual_clean_col not in df.columns:
                raise ValueError(f"Sheet '{sheet_name}' must contain column: {manual_clean_col}")
            self.benchmark_data[sheet_name] = df
    
    def evaluate_source(self, source_name: str, cleaned_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate cleaned texts for a specific source against its benchmark data.
        
        Args:
            source_name (str): Name of the source/sheet
            cleaned_df (pd.DataFrame): DataFrame containing cleaned texts
            
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
        """
        if source_name not in self.benchmark_data:
            raise ValueError(f"Source '{source_name}' not found in benchmark data")
            
        if self.cleaned_text_col not in cleaned_df.columns:
            raise ValueError(f"Cleaned dataframe must contain column: {self.cleaned_text_col}")
            
        benchmark_df = self.benchmark_data[source_name]
        
        # Calculate Levenshtein distances
        distances = []
        normalized_distances = []
        
        for idx, row in benchmark_df.iterrows():
            if idx >= len(cleaned_df):
                break
                
            manual_clean = str(row[self.manual_clean_col])
            llm_clean = str(cleaned_df.iloc[idx][self.cleaned_text_col])
            
            # Calculate Levenshtein distance
            lev_distance = distance(manual_clean, llm_clean)
            distances.append(lev_distance)
            
            # Normalize distance (0-1 scale)
            max_len = max(len(manual_clean), len(llm_clean))
            if max_len == 0:
                normalized_distances.append(0.0)
            else:
                normalized_distances.append(lev_distance / max_len)
        
        # Add normalized distances to the cleaned dataframe
        cleaned_df['normalized_levenshtein'] = normalized_distances
        
        # Calculate and store metrics
        metrics = {
            'mean_distance': np.mean(distances),
            'mean_normalized_distance': np.mean(normalized_distances),
            'std_distance': np.std(distances),
            'std_normalized_distance': np.std(normalized_distances)
        }
        
        self.results[source_name] = metrics
        return metrics
    
    def evaluate_all_sources(self, source_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all sources against their corresponding cleaned dataframes.
        
        Args:
            source_dfs (Dict[str, pd.DataFrame]): Dictionary mapping source names to their cleaned dataframes
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing evaluation metrics for each source
        """
        for source_name, df in source_dfs.items():
            self.evaluate_source(source_name, df)
        
        return self.results
    
    def save_results(self, output_file: str):
        """
        Save evaluation results to a JSON file.
        
        Args:
            output_file (str): Path to save the results
        """
        # Create a simplified version with just the mean normalized distances
        simplified_results = {
            source: metrics['mean_normalized_distance']
            for source, metrics in self.results.items()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, indent=4) 