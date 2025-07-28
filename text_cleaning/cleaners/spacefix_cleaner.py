import pandas as pd
import re
from transformers import pipeline
from .base_cleaner import BaseCleaner
import os
class SpaceFixCleaner(BaseCleaner):
    _oracle = None  # Class variable to hold the pipeline singleton
    _insertions_df = None  # Class variable to hold all insertions DataFrame

    @classmethod
    def get_oracle(cls):
        if cls._oracle is None:
            cls._oracle = pipeline('token-classification', model='dicta-il/dictabert-char-spacefix')
        return cls._oracle

    def __init__(self):
        super().__init__()
        # Do not load the pipeline here; use get_oracle when needed
        # Hebrew Unicode range: \u0590-\u05FF
        self.hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
        self.sp_tag = ' '
        self.threshold = 0.9

    def _restore_spaces_with_tracking(self, text: str):
        if not isinstance(text, str) or not text.strip():
            return text
        oracle = self.get_oracle()
        raw_output = oracle(text)
        new_text = ''
        try:
            assert len(raw_output) == len(text)
        except AssertionError:
            print(f"AssertionError: the model calculated a different length than the text: {text}")
            return text
        for i, (o, c) in enumerate(zip(raw_output,text)):
            is_heb = self.hebrew_pattern.match(o['word'])
            is_heb_minus_1 = self.hebrew_pattern.match(raw_output[i-1]['word']) if i > 0 else False
            if o['entity'] == 'LABEL_1' and o['score'] > self.threshold and is_heb and is_heb_minus_1:
                new_text += f'{self.sp_tag}{c}'
            else:
                new_text += c
        try:
            assert len(new_text.replace(' ', '')) == len(text.replace(' ', ''))
            return new_text
        except AssertionError:
            print(f"AssertionError: the model added text to the text: {text}")
            return text

    def _clean_implementation(self, df: pd.DataFrame, file_name: str = "unknown") -> pd.DataFrame:
        if 'text' not in df.columns:
            return df
        df = df.copy()
        all_insertions = []
        def process_and_track(text):
            new_text = self._restore_spaces_with_tracking(text)
            return new_text
        df['text'] = df['text'].apply(process_and_track)
        if all_insertions:
            # Add dataset name to the insertions data
            for insertion in all_insertions:
                insertion['dataset_name'] = file_name
            
            # Check if file exists and append or create new
            if os.path.exists('spacefix_insertions.csv'):
                # Read existing file and append new data
                existing_df = pd.read_csv('spacefix_insertions.csv')
                new_df = pd.DataFrame(all_insertions)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_csv('spacefix_insertions.csv', index=False)
            else:
                # Create new file
                pd.DataFrame(all_insertions).to_csv('spacefix_insertions.csv', index=False)
        return df 

    def apply_insertions_to_text(self, original_text, insertions):
        """
        Apply insertions to the original_text, skipping those with should_ignore=True.
        Adjusts subsequent indexes if a char is inserted.
        Args:
            original_text (str): The original text.
            insertions (list): List of dicts with 'char_index', 'char', 'should_ignore'.
        Returns:
            str: The modified text after applying insertions.
        """
        # Sort insertions by char_index
        insertions = sorted(insertions, key=lambda x: x['char_index'])
        text = list(original_text)
        offset = 0
        for ins in insertions:
            if ins.get('should_ignore', False):
                continue
            idx = ins['char_index'] + offset
            if idx < 0:
                idx = 0
            if idx > len(text):
                idx = len(text)
            text.insert(idx, ins['char'])
            offset += 1
        return ''.join(text) 