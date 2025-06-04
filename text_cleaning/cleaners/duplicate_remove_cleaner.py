from .base_cleaner import BaseCleaner
import pandas as pd

class DuplicateRemoverCleaner(BaseCleaner):
    def __init__(self):
        ...

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate lines from the 'text' column of the DataFrame.
        """
        res_df = pd.DataFrame(columns=['text', 'n_count'])
        new_texts = []
        new_n_count = []
        for text in df['text']:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            lines = set(lines)
            new_text = ('\n').join(lines)
            new_n_count.append(len(new_text.split()))
            new_texts.append(new_text)
        res_df['text'] = new_texts
        res_df['n_count'] = new_n_count
        return res_df


    def clean(self, df: pd.DataFrame) -> str:
        return self.clean_dataframe(df)
