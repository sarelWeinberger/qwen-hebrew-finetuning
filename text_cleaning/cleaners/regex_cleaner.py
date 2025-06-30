from .base_cleaner import BaseCleaner
import pandas as pd
import regex
import time
import boto3
import io
import os
from utils.logger import logger
from utils.cleaner_constants import CLEANUP_RULES

class RegExCleaner(BaseCleaner):
    def __init__(self, patterns: list[tuple[str, str]] = None, debug_mode: bool = False, save_cleaned_data: bool = False):
        super().__init__()
        # Handle both string and callable replacements
        self.patterns = []
        for p, r in patterns or []:
            if callable(r):
                # For callable replacements, we need to handle them differently
                self.patterns.append((regex.compile(p), r))
            else:
                self.patterns.append((regex.compile(p), r))
        
        self.debug_mode = debug_mode
        self.save_cleaned_data = save_cleaned_data
        self.s3_client = boto3.client('s3') if debug_mode else None
        
        # Initialize examples tracking for each regex pattern from CLEANUP_RULES
        self.examples_collected = {rule['regex'][0]: 0 for rule in CLEANUP_RULES} if debug_mode else {}
        self.examples_data = {rule['regex'][0]: [] for rule in CLEANUP_RULES} if debug_mode else {}
        self.should_stop_processing = False
        
        # Create local directory for storing examples temporarily
        if debug_mode:
            self.local_examples_dir = "regex_examples_temp"
            os.makedirs(self.local_examples_dir, exist_ok=True)
        
        logger.info(f"Initialized RegExCleaner with debug_mode={debug_mode}, save_cleaned_data={save_cleaned_data}")

    def _save_examples_locally(self, regex_pattern: str, examples_df: pd.DataFrame):
        """
        Save examples DataFrame locally until we reach 50 examples.
        
        Args:
            regex_pattern: The regex pattern that was applied
            examples_df: DataFrame with before/after examples
        """
        try:
            # Create a safe filename from the regex pattern
            safe_pattern = regex_pattern.replace('/', '_').replace('\\', '_').replace('*', '_').replace('?', '_')
            safe_pattern = safe_pattern[:50]  # Limit length
            
            # Generate filename
            filename = f"regex_examples_{safe_pattern}.csv"
            filepath = os.path.join(self.local_examples_dir, filename)
            
            # Save to local file
            examples_df.to_csv(filepath, index=False, encoding='utf-8')
            
            logger.info(f"Saved {len(examples_df)} examples locally to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving examples locally: {str(e)}")

    def _save_examples_to_s3(self, rule_info: dict, examples_df: pd.DataFrame):
        """
        Save examples DataFrame to S3 bucket and path specified in the rule.
        
        Args:
            rule_info: Dictionary containing bucket_name, path, and other rule information
            examples_df: DataFrame with before/after examples
        """
        try:
            bucket_name = rule_info['bucket_name']
            path = rule_info['path']
            regex_pattern = rule_info['regex'][0]
            
            # Create a safe filename from the regex pattern
            safe_pattern = regex_pattern.replace('/', '_').replace('\\', '_').replace('*', '_').replace('?', '_')
            safe_pattern = safe_pattern[:50]  # Limit length
            
            # Generate filename with timestamp
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"regex_examples_{safe_pattern}_{timestamp}.csv"
            
            # Create full S3 key
            s3_key = f"{path.rstrip('/')}/{filename}"
            
            # Prepare CSV data
            csv_buffer = io.StringIO()
            examples_df.to_csv(csv_buffer, index=False, encoding='utf-8')
            csv_data = csv_buffer.getvalue()
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=csv_data,
                ContentType='text/csv'
            )
            
            logger.info(f"Saved {len(examples_df)} examples to s3://{bucket_name}/{s3_key}")
            
            # Remove local file after successful S3 upload
            local_filename = f"regex_examples_{safe_pattern}.csv"
            local_filepath = os.path.join(self.local_examples_dir, local_filename)
            if os.path.exists(local_filepath):
                os.remove(local_filepath)
                logger.info(f"Removed local file: {local_filepath}")
            
        except Exception as e:
            logger.error(f"Error saving examples to S3: {str(e)}")

    def _collect_example(self, original_text: str, cleaned_text: str, regex_pattern: str, rule_info: dict):
        """
        Collect an example for a specific regex pattern if in debug mode.
        
        Args:
            original_text: Original text before cleaning
            cleaned_text: Text after cleaning
            regex_pattern: The regex pattern that was applied
            rule_info: Dictionary containing rule information
        """
        if not self.debug_mode or regex_pattern not in self.examples_collected:
            return
        
        # Only collect if the text actually changed
        if original_text != cleaned_text:
            # Check if we still need examples for this pattern
            if self.examples_collected[regex_pattern] < 50:
                example = {
                    'original_text': original_text,
                    'cleaned_text': cleaned_text,
                    'regex_pattern': regex_pattern,
                    'rule_info': rule_info['info'],
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                self.examples_data[regex_pattern].append(example)
                self.examples_collected[regex_pattern] += 1
                
                logger.info(f"Collected example {self.examples_collected[regex_pattern]}/50 for pattern: {regex_pattern[:50]}...")
                
                # Save locally after each example collection
                examples_df = pd.DataFrame(self.examples_data[regex_pattern])
                self._save_examples_locally(regex_pattern, examples_df)
                
                # If we've collected 50 examples for this pattern, save them to S3
                if self.examples_collected[regex_pattern] == 50:
                    self._save_examples_to_s3(rule_info, examples_df)
                    
                    # Check if we've collected enough examples for all patterns
                    if all(count >= 50 for count in self.examples_collected.values()):
                        logger.info("Collected 50 examples for all patterns. Will stop processing after current batch.")
                        self.should_stop_processing = True

    def _clean_implementation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame using the defined regex patterns.
        
        Args:
            df: Input DataFrame with 'text' column
            
        Returns:
            Cleaned DataFrame with 'text' and 'n_words' columns
        """
        if self.should_stop_processing:
            logger.info("Debug mode: Stopping processing as all examples have been collected.")
            return df

        cleaned_texts = []
        n_words = []

        _DELIM = "UNIQUE_DELIMITER_XYZ123_456_789_899_234_123"

        # 1. Join all rows into one long string
        joined_text = _DELIM.join(df["text"].astype(str).tolist())
        original_texts = df["text"].astype(str).tolist()

        # 2. Apply every (pattern → replacement) once over the *entire* string
        for pattern, repl in self.patterns:
            if self.should_stop_processing:
                break
                
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
                
                # If in debug mode, collect examples for this pattern
                if self.debug_mode:
                    # Find the corresponding rule info from CLEANUP_RULES
                    rule_info = None
                    for rule in CLEANUP_RULES:
                        if rule['regex'][0] == pattern.pattern:
                            rule_info = rule
                            break
                    
                    if rule_info:
                        # Split back to individual texts to collect examples
                        if callable(repl):
                            # We already have the before/after texts
                            texts_before_split = texts_before
                            texts_after_split = texts_after
                        else:
                            texts_before_split = text_before_pattern.split(_DELIM)
                            texts_after_split = joined_text.split(_DELIM)
                        
                        # Collect examples for this pattern
                        for i, (orig, cleaned) in enumerate(zip(texts_before_split, texts_after_split)):
                            if self.should_stop_processing:
                                break
                            if orig != cleaned:
                                self._collect_example(orig, cleaned, pattern.pattern, rule_info)
                                if self.should_stop_processing:
                                    break

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
        
        # If save_cleaned_data is True, save the cleaned data
        if self.save_cleaned_data:
            self._save_cleaned_data(cleaned_df)
        
        # If in debug mode and we've collected examples for all patterns, stop processing
        if self.debug_mode and self.should_stop_processing:
            logger.info("Debug mode: Collected 50 examples for all patterns. Stopping processing.")
            
        return cleaned_df

    def _save_cleaned_data(self, cleaned_df: pd.DataFrame):
        """
        Save the cleaned data to a local file.
        
        Args:
            cleaned_df: DataFrame with cleaned data
        """
        try:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"cleaned_data_{timestamp}.csv"
            filepath = os.path.join(self.local_examples_dir, filename)
            
            cleaned_df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"Saved cleaned data to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving cleaned data: {str(e)}")

    def get_debug_stats(self) -> dict:
        """
        Get debug mode statistics.
        
        Returns:
            Dictionary with debug statistics
        """
        if not self.debug_mode:
            return {}
        
        return {
            'examples_collected': self.examples_collected,
            'total_examples': sum(self.examples_collected.values()),
            'patterns_with_50_examples': sum(1 for count in self.examples_collected.values() if count >= 50),
            'should_stop_processing': self.should_stop_processing,
            'save_cleaned_data': self.save_cleaned_data
        }