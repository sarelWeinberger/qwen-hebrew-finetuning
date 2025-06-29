from .base_cleaner import BaseCleaner
import pandas as pd
import time
from utils.logger import logger


class DuplicateRemoverCleaner(BaseCleaner):
    def __init__(self, save_samples: bool = True, sample_percentage: float = 0.05):
        super().__init__(save_samples=save_samples, sample_percentage=sample_percentage)
        logger.info("Initialized DuplicateRemoverCleaner")

    def _clean_implementation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate lines from the 'text' column of the DataFrame.
        
        Args:
            df: Input DataFrame with 'text' column
            
        Returns:
            Cleaned DataFrame with 'text' and 'n_words' columns
        """
        new_texts = []
        new_n_words = []
        
        total_lines_removed = 0
        total_original_lines = 0
        
        for idx, text in enumerate(df['text']):
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            total_original_lines += len(lines)
            seen = set()
            unique_lines = []
            
            for line in lines:
                if line not in seen:
                    seen.add(line)
                    unique_lines.append(line)
                else:
                    total_lines_removed += 1
            
            new_text = '\n'.join(unique_lines)
            new_n_words.append(len(new_text.split()))
            new_texts.append(new_text)
            
            if len(unique_lines) != len(lines):
                self.stats['rows_modified'] += 1

        # Create result DataFrame
        result_df = pd.DataFrame({
            'text': new_texts,
            'n_words': new_n_words
        })
        
        # Remove duplicate rows
        result_df = result_df.drop_duplicates()
        
        # Update statistics
        self.stats['patterns_matched'] = {
            'duplicate_lines': total_lines_removed,
            'total_original_lines': total_original_lines
        }
        
        # Log detailed statistics
        logger.info(f"DuplicateRemoverCleaner processed {len(df)} texts rows")
        logger.info(f"Modified {self.stats['rows_modified']} texts rows")
        logger.info(f"Total lines processed: {total_original_lines}")
        logger.info(f"Duplicate lines removed: {total_lines_removed}")
        if total_original_lines > 0:
            logger.info(f"Duplicate removal rate: {(total_lines_removed/total_original_lines*100):.2f}%")
        
        return result_df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._clean_implementation(df)
