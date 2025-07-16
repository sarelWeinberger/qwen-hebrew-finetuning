import pandas as pd
from transformers import pipeline
from .base_cleaner import BaseCleaner

class SpaceFixCleaner(BaseCleaner):
    _oracle = None  # Class variable to hold the pipeline singleton

    @classmethod
    def get_oracle(cls):
        if cls._oracle is None:
            cls._oracle = pipeline('token-classification', model='dicta-il/dictabert-char-spacefix')
        return cls._oracle

    def __init__(self):
        super().__init__()
        # Do not load the pipeline here; use get_oracle when needed

    def _restore_spaces_with_tracking(self, text: str):
        if not isinstance(text, str) or not text.strip():
            return text, []
        oracle = self.get_oracle()
        raw_output = oracle(text)
        new_text = []
        insertions = []
        for i, o in enumerate(raw_output):
            if o['entity'] == 'LABEL_1':
                insertions.append({
                    'original_text': text,
                    'modified_text': None,
                    'char_index': i,
                    'char': o['word']
                })
                new_text.append(' ')
            new_text.append(o['word'])
        result = ''.join(new_text)
        for ins in insertions:
            ins['modified_text'] = result
        return result, insertions

    def _clean_implementation(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'text' not in df.columns:
            return df
        df = df.copy()
        all_insertions = []
        def process_and_track(text):
            after, insertions = self._restore_spaces_with_tracking(text)
            all_insertions.extend(insertions)
            return after
        df['text'] = df['text'].apply(process_and_track)
        if all_insertions:
            import pandas as pd
            pd.DataFrame(all_insertions).to_csv('spacefix_insertions.csv', index=False)
        return df 