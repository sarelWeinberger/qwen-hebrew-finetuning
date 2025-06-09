import pandas as pd
import numpy as np
from Levenshtein import distance
from typing import List, Tuple, Dict
import json
from pathlib import Path


class BenchmarkEvaluator:
    def __init__(self, benchmark_file: str):
        """
        Initialize the benchmark evaluator with a benchmark file.
        
        Args:
            benchmark_file (str): Path to the benchmark file (CSV) containing raw and cleaned text pairs
        """
        self.benchmark_data = pd.read_csv(benchmark_file)
        self.required_columns = ['raw_text', 'cleaned_text']
        
        # Validate benchmark file
        if not all(col in self.benchmark_data.columns for col in self.required_columns):
            raise ValueError(f"Benchmark file must contain columns: {self.required_columns}")
    
    def evaluate_texts(self, texts: List[str], cleaned_texts: List[str]) -> Dict[str, List[float]]:
        """
        Evaluate a list of texts against their cleaned versions using Levenshtein distance.
        
        Args:
            texts (List[str]): List of raw texts
            cleaned_texts (List[str]): List of corresponding cleaned texts
            
        Returns:
            Dict[str, List[float]]: Dictionary containing:
                - 'distances': List of Levenshtein distances
                - 'normalized_distances': List of normalized Levenshtein distances (0-1)
        """
        if len(texts) != len(cleaned_texts):
            raise ValueError("Number of texts and cleaned texts must be equal")
        
        distances = []
        normalized_distances = []
        
        for raw, cleaned in zip(texts, cleaned_texts):
            # Calculate Levenshtein distance
            lev_distance = distance(raw, cleaned)
            distances.append(lev_distance)
            
            # Normalize distance (0-1 scale)
            max_len = max(len(raw), len(cleaned))
            if max_len == 0:
                normalized_distances.append(0.0)
            else:
                normalized_distances.append(lev_distance / max_len)
        
        return {
            'distances': distances,
            'normalized_distances': normalized_distances
        }
    
    def evaluate_benchmark(self, cleaned_texts: List[str]) -> Dict[str, float]:
        """
        Evaluate cleaned texts against the benchmark data.
        
        Args:
            cleaned_texts (List[str]): List of cleaned texts to evaluate
            
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics:
                - 'mean_distance': Average Levenshtein distance
                - 'mean_normalized_distance': Average normalized distance
                - 'std_distance': Standard deviation of distances
                - 'std_normalized_distance': Standard deviation of normalized distances
        """
        if len(cleaned_texts) != len(self.benchmark_data):
            raise ValueError("Number of cleaned texts must match benchmark data length")
        
        results = self.evaluate_texts(
            self.benchmark_data['raw_text'].tolist(),
            cleaned_texts
        )
        
        return {
            'mean_distance': np.mean(results['distances']),
            'mean_normalized_distance': np.mean(results['normalized_distances']),
            'std_distance': np.std(results['distances']),
            'std_normalized_distance': np.std(results['normalized_distances'])
        }
    
    def save_results(self, results: Dict[str, float], output_file: str):
        """
        Save evaluation results to a JSON file.
        
        Args:
            results (Dict[str, float]): Evaluation results
            output_file (str): Path to save the results
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4) 