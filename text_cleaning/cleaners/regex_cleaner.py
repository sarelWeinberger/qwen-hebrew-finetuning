from .base_cleaner import BaseCleaner
import pandas as pd
import regex
import time
from utils.logger import logger

class RegExCleaner(BaseCleaner):
    def __init__(self, patterns: list[tuple[str, str]] = None, save_samples: bool = True, sample_percentage: float = 0.05):
        super().__init__(save_samples=save_samples, sample_percentage=sample_percentage)
        self.patterns = [(regex.compile(p), r) for p, r in patterns or []]
        logger.info(f"Initialized RegExCleaner with {len(self.patterns)} patterns")

    def _clean_implementation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame using the defined regex patterns.
        
        Args:
            df: Input DataFrame with 'text' column
            
        Returns:
            Cleaned DataFrame with 'text' and 'n_words' columns
        """
        cleaned_texts = []
        n_words = []

        _DELIM = "UNIQUE_DELIMITER_XYZ123_456_789_899_234_123"

        # 1. Join all rows into one long string
        joined_text = _DELIM.join(df["text"].astype(str).tolist())

        # 2. Apply every (pattern → replacement) once over the *entire* string
        for pattern, repl in self.patterns:
            joined_text, n_subs = pattern.subn(repl, joined_text)
            if n_subs:  # update stats only when we actually replaced something
                self.stats["patterns_matched"][pattern.pattern] = (
                    self.stats["patterns_matched"].get(pattern.pattern, 0) + n_subs
                )

        # 3. Split back to the original rows and final post-processing
        cleaned_texts = [t.strip() for t in joined_text.split(_DELIM)]
        n_words = [len(t.split()) for t in cleaned_texts]

        # Create result DataFrame
        result_df = pd.DataFrame({
            "text": cleaned_texts,
            "n_words": n_words
        })
        
        # Calculate rows modified
        original_texts = df["text"].astype(str).tolist()
        modified_count = sum(1 for orig, cleaned in zip(original_texts, cleaned_texts) if orig != cleaned)
        self.stats['rows_modified'] = modified_count
        
        logger.info(f"RegExCleaner processed {len(df)} rows")
        logger.info(f"Modified {self.stats['rows_modified']} rows")

        for pattern, count in self.stats['patterns_matched'].items():
            logger.info(f"Pattern '{pattern}' matched {count} times")

        return result_df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame using the defined regex patterns.
        """        
        cleaned_df = self._clean_implementation(df)
        return cleaned_df