from .base_cleaner import BaseCleaner
import pandas as pd
import re


class RegExCleaner(BaseCleaner):
    def __init__(self, patterns: list[tuple[str, str]] = None):
        """
        :param patterns: A list of (pattern, replacement) tuples.
                         Example: [(r'<[^>]+>', ''), (r'\s+', ' ')]
        """
        self.patterns = patterns or []

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply regex replacements on the 'text' column.
        """
        cleaned_texts = []
        n_words = []

        for text in df['text']:
            for pattern, repl in self.patterns:
                text = re.sub(pattern, repl, text)
            text = text.strip()
            cleaned_texts.append(text)
            n_words.append(len(text.split()))

        return pd.DataFrame({'text': cleaned_texts, 'n_count': n_words})

    def clean(self, df: pd.DataFrame) -> str:
        return self.clean_dataframe(df)