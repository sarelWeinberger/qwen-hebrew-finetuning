from .base_cleaner import BaseCleaner
import pandas as pd
import re
import time
from utils.logger import logger


class RegExCleaner(BaseCleaner):
    def __init__(self, patterns: list[tuple[str, str]] = None):
        """
        :param patterns: A list of (pattern, replacement) tuples.
                         Example: [(r'<[^>]+>', ''), (r'\s+', ' ')]
        """
        super().__init__()
        self.patterns = patterns or []
        logger.info(f"Initialized RegExCleaner with {len(self.patterns)} patterns")
        for pattern, repl in self.patterns:
            logger.debug(f"Pattern: '{pattern}' -> Replacement: '{repl}'")

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply regex replacements on the 'text' column.
        """
        start_time = time.time()
        cleaned_texts = []
        n_words = []
        
        self.stats['total_rows_processed'] = len(df)
        original_lengths = df['text'].str.len()
        
        for idx, text in enumerate(df['text']):
            original_text = text
            original_len = len(text)
            
            for pattern, repl in self.patterns:
                matches = re.findall(pattern, text)
                if matches:
                    self.stats['patterns_matched'][pattern] = self.stats['patterns_matched'].get(pattern, 0) + len(matches)
                    text = re.sub(pattern, repl, text)
            
            text = text.strip()
            cleaned_texts.append(text)
            n_words.append(len(text.split()))
            
            # Update statistics
            new_len = len(text)
            if new_len != original_len:
                self.stats['rows_modified'] += 1
                self.stats['characters_removed'] += max(0, original_len - new_len)
                self.stats['characters_added'] += max(0, new_len - original_len)
        
        self.stats['execution_time'] = time.time() - start_time
        
        # Log detailed statistics
        logger.info(f"Processed {len(df)} rows")
        logger.info(f"Modified {self.stats['rows_modified']} rows")
        logger.info(f"Total characters removed: {self.stats['characters_removed']}")
        logger.info(f"Total characters added: {self.stats['characters_added']}")
        
        for pattern, count in self.stats['patterns_matched'].items():
            logger.info(f"Pattern '{pattern}' matched {count} times")
        
        return pd.DataFrame({'text': cleaned_texts, 'n_count': n_words})

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.clean_dataframe(df)