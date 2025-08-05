from .base_cleaner import BaseCleaner
import pandas as pd
import regex
import time
import os

class RegExCleaner(BaseCleaner):
    def __init__(self, patterns: list[tuple[str, str]] = None, save_word_changes: bool = True):
        super().__init__()
        # Handle both string and callable replacements
        self.patterns = []
        for p, r in patterns or []:
            if callable(r):
                # For callable replacements, we need to handle them differently
                self.patterns.append((regex.compile(p), r))
            else:
                self.patterns.append((regex.compile(p), r))
        
        self.save_word_changes = save_word_changes
        self.word_changes = []
        
        # Create directory for saving files
        if save_word_changes:
            self.output_dir = "regex_examples_temp"
            os.makedirs(self.output_dir, exist_ok=True)

    def _track_word_changes(self, original_text: str, cleaned_text: str, pattern: str):
        """
        Track word-level changes between original and cleaned text.
        """
        if not self.save_word_changes:
            return
            
        # Split texts into words
        original_words = original_text.split()
        cleaned_words = cleaned_text.split()
        
        # Find differences between word lists
        max_len = max(len(original_words), len(cleaned_words))
        
        for i in range(max_len):
            original_word = original_words[i] if i < len(original_words) else ""
            cleaned_word = cleaned_words[i] if i < len(cleaned_words) else ""
            
            # If words are different, track the change
            if original_word != cleaned_word:
                self.word_changes.append({
                    'before': original_word,
                    'after': cleaned_word,
                    'regex_pattern': pattern
                })

    def save_word_changes_to_file(self):
        """
        Save word changes to CSV file.
        """
        if not self.save_word_changes or not self.word_changes:
            return
            
        try:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"word_changes_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            df = pd.DataFrame(self.word_changes)
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            print(f"Saved {len(df)} word changes to {filepath}")
            
        except Exception as e:
            print(f"Error saving word changes: {str(e)}")

    def get_word_changes_df(self) -> pd.DataFrame:
        """
        Get DataFrame of word changes.
        """
        return pd.DataFrame(self.word_changes)

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
        original_texts = df["text"].astype(str).tolist()

        # 2. Apply every (pattern → replacement) once over the *entire* string
        for pattern, repl in self.patterns:
            # Store original text before applying this pattern
            text_before_pattern = joined_text
            
            # Handle callable replacements differently
            if callable(repl):
                # For callable replacements, we need to apply them differently
                # This is more complex and we'll need to process each text individually
                texts_before = text_before_pattern.split(_DELIM)
                texts_after = []
                total_subs = 0
                
                for text in texts_before:
                    new_text, n_subs = pattern.subn(repl, text)
                    texts_after.append(new_text)
                    total_subs += n_subs
                
                joined_text = _DELIM.join(texts_after)
                n_subs = total_subs
            else:
                # Regular string replacement
                joined_text, n_subs = pattern.subn(repl, joined_text)
            
            if n_subs:  # update stats only when we actually replaced something
                self.stats["patterns_matched"][pattern.pattern] = (
                    self.stats["patterns_matched"].get(pattern.pattern, 0) + n_subs
                )
                
                # Track word changes if enabled
                if self.save_word_changes:
                    # Split back to individual texts to collect examples
                    if callable(repl):
                        # We already have the before/after texts
                        texts_before_split = texts_before
                        texts_after_split = texts_after
                    else:
                        texts_before_split = text_before_pattern.split(_DELIM)
                        texts_after_split = joined_text.split(_DELIM)
                    
                    # Collect examples for this pattern
                    for orig, cleaned in zip(texts_before_split, texts_after_split):
                        if orig != cleaned:
                            self._track_word_changes(orig, cleaned, pattern.pattern)

        # 3. Split back to the original rows and final post-processing
        cleaned_texts = [t.strip() for t in joined_text.split(_DELIM)]
        n_words = [len(t.split()) for t in cleaned_texts]

        # Create result DataFrame
        result_df = pd.DataFrame({
            "text": cleaned_texts,
            "n_words": n_words
        })
        
        # Calculate rows modified
        modified_count = sum(1 for orig, cleaned in zip(original_texts, cleaned_texts) if orig != cleaned)
        self.stats['rows_modified'] = modified_count

        return result_df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame using the defined regex patterns.
        """        
        cleaned_df = self._clean_implementation(df)
        
        # Save word changes if enabled
        if self.save_word_changes:
            self.save_word_changes_to_file()
            
        return cleaned_df

    def get_stats(self) -> dict:
        """
        Get cleaning statistics.
        
        Returns:
            Dictionary with cleaning statistics
        """
        stats = {
            'patterns_matched': self.stats.get('patterns_matched', {}),
            'rows_modified': self.stats.get('rows_modified', 0)
        }
        
        if self.save_word_changes:
            stats['word_changes_tracked'] = len(self.word_changes)
        
        return stats