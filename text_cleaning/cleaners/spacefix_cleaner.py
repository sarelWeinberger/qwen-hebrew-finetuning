import pandas as pd
import re
from transformers import pipeline
from .base_cleaner import BaseCleaner
import os
import boto3
class SpaceFixCleaner(BaseCleaner):
    _oracle = None  # Class variable to hold the pipeline singleton
    _insertions_df = None  # Class variable to hold all insertions DataFrame
    _tracking_enabled = False  # Class variable to enable/disable tracking
    _tracking_data = []  # Class variable to store tracking data

    @classmethod
    def get_oracle(cls):
        if cls._oracle is None:
            # Check if CUDA is available and use GPU if possible
            import torch
            device = 0 if torch.cuda.is_available() else -1
            cls._oracle = pipeline('token-classification', 
                                 model='dicta-il/dictabert-char-spacefix',
                                 device=device)
            print(f"SpaceFixCleaner initialized on device: {'GPU' if device == 0 else 'CPU'}")
        return cls._oracle

    def __init__(self, enable_tracking=False):
        super().__init__()
        # Do not load the pipeline here; use get_oracle when needed
        # Hebrew Unicode range: \u0590-\u05FF
        self.hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
        self.sp_tag = ' '
        self.threshold = 0.9
        self._tracking_enabled = enable_tracking
        if enable_tracking:
            self._tracking_data = []

    def enable_tracking(self):
        """Enable tracking of before/after data during cleaning."""
        self._tracking_enabled = True
        self._tracking_data = []

    def disable_tracking(self):
        """Disable tracking of before/after data during cleaning."""
        self._tracking_enabled = False
        self._tracking_data = []

    def get_tracking_data(self):
        """Get the collected tracking data as a DataFrame."""
        if not self._tracking_data:
            return pd.DataFrame()
        return pd.DataFrame(self._tracking_data)

    def clear_tracking_data(self):
        """Clear the collected tracking data."""
        self._tracking_data = []

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
            original_text = text
            
            # Apply space fixer
            new_text = self._restore_spaces_with_tracking(text)
            
            # Track data if enabled
            if self._tracking_enabled:
                # Calculate basic statistics
                original_spaces = original_text.count(' ')
                fixed_spaces = new_text.count(' ')
                spaces_added = fixed_spaces - original_spaces
                
                # Calculate word counts (simple split by spaces)
                original_words = len(original_text.split())
                fixed_words = len(new_text.split())
                
                # Store tracking data
                self._tracking_data.append({
                    'original_text': original_text,
                    'fixed_text': new_text,
                    'original_spaces': original_spaces,
                    'fixed_spaces': fixed_spaces,
                    'spaces_added': spaces_added,
                    'original_word_count': original_words,
                    'fixed_word_count': fixed_words,
                    'word_count_change': fixed_words - original_words,
                    'has_changes': original_text != new_text,
                    'file_name': file_name
                })
            
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

    def count_words_before_after(self, source_name, output_bucket_name, output_prefix):
        """Count words before and after space fixing and save results."""
        try:
            # Count words in the tracking data
            if not self._tracking_data:
                print(f"No tracking data available for {source_name}")
                return
            
            tracking_df = pd.DataFrame(self._tracking_data)
            
            # Calculate total words before and after
            total_words_before = tracking_df['original_word_count'].sum()
            total_words_after = tracking_df['fixed_word_count'].sum()
            
            # Create simple results summary
            results_summary = f"""n words before cleaning: {total_words_before:,}
n words after cleaning: {total_words_after:,}"""
            
            # Save results to S3
            self.save_word_count_results(results_summary, source_name, output_bucket_name, output_prefix)
            
            print(f"Word count analysis completed for {source_name}")
            print(f"Raw: {total_words_before:,} words, Cleaned: {total_words_after:,} words")
            
        except Exception as e:
            print(f"Error in word count analysis: {str(e)}")
            # Save error message
            error_summary = f"""n words before cleaning: 0
n words after cleaning: 0
Error: {str(e)}"""
            self.save_word_count_results(error_summary, source_name, output_bucket_name, output_prefix)

    def save_word_count_results(self, results_text, source_name, output_bucket_name, output_prefix):
        """Save word count results as a text file to S3."""
        try:
            s3 = boto3.client("s3")
            
            # Create simple filename
            filename = f"word_count_{source_name}.txt"
            
            # Create the output key
            output_key = f"{output_prefix.rstrip('/')}/{filename}"
            
            # Upload to S3
            s3.put_object(
                Bucket=output_bucket_name,
                Key=output_key,
                Body=results_text.encode('utf-8'),
                ContentType='text/plain'
            )
            
            print(f"Word count results saved to s3://{output_bucket_name}/{output_key}")
            
        except Exception as e:
            print(f"Error saving word count results: {str(e)}")

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