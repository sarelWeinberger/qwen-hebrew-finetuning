"""
Sample Saver Utility

This module provides functionality to save before and after samples
of the cleaning process for analysis and quality control.
"""

import pandas as pd
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import random
from utils.logger import logger


class SampleSaver:
    def __init__(self, output_dir: str = "text_cleaning/samples", sample_percentage: float = 0.05):
        """
        Initialize the sample saver.
        
        Args:
            output_dir: Directory to save samples
            sample_percentage: Percentage of data to sample (0.0 to 1.0)
        """
        self.output_dir = Path(output_dir)
        self.sample_percentage = sample_percentage
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.before_dir = self.output_dir / "before"
        self.after_dir = self.output_dir / "after"
        self.comparison_dir = self.output_dir / "comparison"
        
        for dir_path in [self.before_dir, self.after_dir, self.comparison_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def select_sample_indices(self, total_rows: int) -> List[int]:
        """
        Select random indices for sampling.
        
        Args:
            total_rows: Total number of rows in the dataset
            
        Returns:
            List of selected indices
        """
        sample_size = max(1, int(total_rows * self.sample_percentage))
        sample_size = min(sample_size, total_rows)  # Don't sample more than available
        
        indices = list(range(total_rows))
        selected_indices = random.sample(indices, sample_size)
        selected_indices.sort()  # Keep them in order for easier comparison
        
        return selected_indices
    
    def save_samples(self, 
                    before_df: pd.DataFrame, 
                    after_df: pd.DataFrame, 
                    cleaner_name: str,
                    file_name: str,
                    metadata: Optional[Dict] = None) -> Dict[str, str]:
        """
        Save before and after samples for comparison.
        
        Args:
            before_df: DataFrame before cleaning
            after_df: DataFrame after cleaning
            cleaner_name: Name of the cleaner for organization
            file_name: Base name for the output files
            metadata: Additional metadata to save
            
        Returns:
            Dictionary with paths to saved files
        """
        if len(before_df) != len(after_df):
            raise ValueError("Before and after DataFrames must have the same number of rows")
        
        # Create timestamp for unique file naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Select sample indices
        sample_indices = self.select_sample_indices(len(before_df))
        
        # Extract samples
        before_sample = before_df.iloc[sample_indices].copy()
        after_sample = after_df.iloc[sample_indices].copy()
        
        # Add index information for tracking
        before_sample['original_index'] = sample_indices
        after_sample['original_index'] = sample_indices
        
        # Create cleaner-specific directories
        cleaner_before_dir = self.before_dir / cleaner_name
        cleaner_after_dir = self.after_dir / cleaner_name
        cleaner_comparison_dir = self.comparison_dir / cleaner_name
        
        for dir_path in [cleaner_before_dir, cleaner_after_dir, cleaner_comparison_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Generate file names
        base_name = f"{file_name}_{timestamp}"
        before_file = cleaner_before_dir / f"{base_name}_before.csv"
        after_file = cleaner_after_dir / f"{base_name}_after.csv"
        comparison_file = cleaner_comparison_dir / f"{base_name}_comparison.csv"
        metadata_file = cleaner_comparison_dir / f"{base_name}_metadata.json"
        
        # Save before sample
        before_sample.to_csv(before_file, index=False, encoding='utf-8')
        
        # Save after sample
        after_sample.to_csv(after_file, index=False, encoding='utf-8')
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'original_index': sample_indices,
            'text_before': before_sample['text'].values,
            'text_after': after_sample['text'].values,
            'n_words_before': before_sample['n_words'].values,
            'n_words_after': after_sample['n_words'].values,
            'text_changed': before_sample['text'] != after_sample['text'],
            'word_count_changed': before_sample['n_words'] != after_sample['n_words']
        })
        
        # Add character count information
        comparison_df['chars_before'] = before_sample['text'].str.len()
        comparison_df['chars_after'] = after_sample['text'].str.len()
        comparison_df['chars_removed'] = comparison_df['chars_before'] - comparison_df['chars_after']
        
        # Save comparison
        comparison_df.to_csv(comparison_file, index=False, encoding='utf-8')
        
        # Save metadata
        metadata_dict = {
            'cleaner_name': cleaner_name,
            'file_name': file_name,
            'timestamp': timestamp,
            'total_rows': len(before_df),
            'sample_size': len(sample_indices),
            'sample_percentage': self.sample_percentage,
            'sample_indices': sample_indices,
            'files_created': {
                'before_sample': str(before_file),
                'after_sample': str(after_file),
                'comparison': str(comparison_file),
                'metadata': str(metadata_file)
            }
        }
        
        if metadata:
            metadata_dict['additional_metadata'] = metadata
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        return {
            'before_sample': str(before_file),
            'after_sample': str(after_file),
            'comparison': str(comparison_file),
            'metadata': str(metadata_file)
        }
    
    def get_sample_summary(self, comparison_file: str) -> Dict:
        """
        Generate a summary of the sample comparison.
        
        Args:
            comparison_file: Path to the comparison CSV file
            
        Returns:
            Dictionary with summary statistics
        """
        comparison_df = pd.read_csv(comparison_file)
        
        total_samples = len(comparison_df)
        text_changed = comparison_df['text_changed'].sum()
        word_count_changed = comparison_df['word_count_changed'].sum()
        
        summary = {
            'total_samples': total_samples,
            'text_changed_count': int(text_changed),
            'text_changed_percentage': float(text_changed / total_samples * 100),
            'word_count_changed_count': int(word_count_changed),
            'word_count_changed_percentage': float(word_count_changed / total_samples * 100),
            'total_chars_removed': int(comparison_df['chars_removed'].sum()),
            'avg_chars_removed_per_sample': float(comparison_df['chars_removed'].mean()),
            'max_chars_removed': int(comparison_df['chars_removed'].max()),
            'min_chars_removed': int(comparison_df['chars_removed'].min())
        }
        
        return summary
    
    def list_samples(self, cleaner_name: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all saved samples.
        
        Args:
            cleaner_name: Optional cleaner name to filter by
            
        Returns:
            Dictionary with lists of sample files
        """
        samples = {
            'before': [],
            'after': [],
            'comparison': []
        }
        
        for sample_type in samples.keys():
            base_dir = getattr(self, f"{sample_type}_dir")
            
            if cleaner_name:
                cleaner_dir = base_dir / cleaner_name
                if cleaner_dir.exists():
                    samples[sample_type] = [str(f) for f in cleaner_dir.glob("*.csv")]
            else:
                for cleaner_dir in base_dir.iterdir():
                    if cleaner_dir.is_dir():
                        samples[sample_type].extend([str(f) for f in cleaner_dir.glob("*.csv")])
        
        return samples 