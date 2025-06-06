from .base_cleaner import BaseCleaner
import pandas as pd
import regex
import time
from utils.logger import logger

class RegExCleaner(BaseCleaner):
    def __init__(self, patterns: list[tuple[str, str]] = None):
        super().__init__()
        self.patterns = [(regex.compile(p), r) for p, r in patterns or []]
        logger.info(f"Initialized RegExCleaner with {len(self.patterns)} patterns")

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        start_time = time.time()
        cleaned_texts = []
        n_words = []

        self.stats['total_rows_processed'] = len(df)

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

        # 4. Push the results back into the dataframe (or return them, as you prefer)
        df["text"] = cleaned_texts
        df["n_count"] = n_words

        self.stats['execution_time'] = time.time() - start_time
        logger.info(f"Processed {len(df)} rows in {self.stats['execution_time']:.2f}s")
        logger.info(f"Modified {self.stats['rows_modified']} rows")

        for pattern, count in self.stats['patterns_matched'].items():
            logger.info(f"Pattern '{pattern}' matched {count} times")

        return pd.DataFrame({'text': cleaned_texts, 'n_count': n_words})
    
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame using the defined regex patterns.
        """        
        cleaned_df = self.clean_dataframe(df)
        return cleaned_df