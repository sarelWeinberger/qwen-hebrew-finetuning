from .base_cleaner import BaseCleaner
import pandas as pd
import re
import time
from utils.logger import logger

class RegExCleaner(BaseCleaner):
    def __init__(self, patterns: list[tuple[str, str]] = None):
        super().__init__()
        self.patterns = [(re.compile(p), r) for p, r in patterns or []]
        logger.info(f"Initialized RegExCleaner with {len(self.patterns)} patterns")

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        start_time = time.time()
        cleaned_texts = []
        n_words = []

        self.stats['total_rows_processed'] = len(df)

        for text in df['text']:
            original_len = len(text)
            modified = False

            for pattern, repl in self.patterns:
                text, n_subs = pattern.subn(repl, text)
                if n_subs > 0:
                    self.stats['patterns_matched'][pattern.pattern] = (
                        self.stats['patterns_matched'].get(pattern.pattern, 0) + n_subs
                    )
                    modified = True

            text = text.strip()
            cleaned_texts.append(text)
            n_words.append(len(text.split()))

            if modified:
                new_len = len(text)
                self.stats['rows_modified'] += 1
                self.stats['characters_removed'] += max(0, original_len - new_len)
                self.stats['characters_added'] += max(0, new_len - original_len)

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