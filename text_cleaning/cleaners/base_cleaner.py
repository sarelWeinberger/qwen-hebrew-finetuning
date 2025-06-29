import pandas as pd
from typing import Dict, Any, Optional
from utils.logger import logger
from utils.sample_saver import SampleSaver
import time

class BaseCleaner:
    def __init__(self, save_samples: bool = True, sample_percentage: float = 0.05, sample_output_dir: str = "text_cleaning/samples"):
        """
        Initialize the base cleaner.
        
        Args:
            save_samples: Whether to save before/after samples
            sample_percentage: Percentage of data to sample (0.0 to 1.0)
            sample_output_dir: Directory to save samples
        """
        self.stats = {
            'total_rows_processed': 0,
            'rows_modified': 0,
            'characters_removed': 0,
            'characters_added': 0,
            'patterns_matched': {},
            'execution_time': 0
        }
        
        # Sample saving configuration
        self.save_samples = save_samples
        self.sample_percentage = sample_percentage
        self.sample_saver = None
        
        if self.save_samples:
            self.sample_saver = SampleSaver(
                output_dir=sample_output_dir,
                sample_percentage=sample_percentage
            )
    
    def clean(self, df: pd.DataFrame, file_name: str = "unknown") -> pd.DataFrame:
        """
        Clean the input DataFrame.
        
        Args:
            df: Input DataFrame to clean
            file_name: Name of the file being processed (for sample naming)
            
        Returns:
            Cleaned DataFrame
        """
        start_time = time.time()
        
        # Save before sample if enabled
        before_df = None
        if self.save_samples and self.sample_saver:
            before_df = df.copy()
        
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
        
        # Save after sample if enabled
        if self.save_samples and self.sample_saver and before_df is not None:
            try:
                sample_files = self.sample_saver.save_samples(
                    before_df=before_df,
                    after_df=cleaned_df,
                    cleaner_name=self.__class__.__name__,
                    file_name=file_name,
                    metadata={
                        'cleaner_stats': self.stats,
                        'sample_percentage': self.sample_percentage
                    }
                )
                
                # Log sample saving results
                
                # Generate and log summary
                summary = self.sample_saver.get_sample_summary(sample_files['comparison'])
                
            except Exception as e:
                logger.warning(f"Failed to save samples for {self.__class__.__name__}: {str(e)}")
        
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
    
    def set_sample_saving(self, enabled: bool, sample_percentage: Optional[float] = None):
        """
        Enable or disable sample saving and optionally change the sample percentage.
        
        Args:
            enabled: Whether to enable sample saving
            sample_percentage: Optional new sample percentage (0.0 to 1.0)
        """
        self.save_samples = enabled
        
        if sample_percentage is not None:
            self.sample_percentage = sample_percentage
        
        if enabled and self.sample_saver is None:
            self.sample_saver = SampleSaver(sample_percentage=self.sample_percentage)
        elif not enabled:
            self.sample_saver = None
    
    def get_sample_files(self) -> Optional[Dict[str, str]]:
        """
        Get the most recent sample files if sample saving is enabled.
        
        Returns:
            Dictionary with sample file paths or None if sample saving is disabled
        """
        if not self.save_samples or self.sample_saver is None:
            return None
        
        # This would need to be implemented to track the most recent sample files
        # For now, return None
        return None