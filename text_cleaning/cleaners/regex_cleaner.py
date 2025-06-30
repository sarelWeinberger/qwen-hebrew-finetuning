from .base_cleaner import BaseCleaner
import pandas as pd
import regex
import time
import boto3
import io
from utils.logger import logger
from utils.cleaner_constants import CLEANUP_RULES

class RegExCleaner(BaseCleaner):
    def __init__(self, patterns: list[tuple[str, str]] = None, save_samples: bool = True, sample_percentage: float = 0.05, debug_mode: bool = False):
        super().__init__(save_samples=save_samples, sample_percentage=sample_percentage)
        self.patterns = [(regex.compile(p), r) for p, r in patterns or []]
        self.debug_mode = debug_mode
        self.s3_client = boto3.client('s3') if debug_mode else None
        self.examples_collected = {rule['regex'][0]: 0 for rule in CLEANUP_RULES} if debug_mode else {}
        self.examples_data = {rule['regex'][0]: [] for rule in CLEANUP_RULES} if debug_mode else {}
        logger.info(f"Initialized RegExCleaner with debug_mode={debug_mode}")

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
                
                # If we've collected 50 examples for this pattern, save them
                if self.examples_collected[regex_pattern] == 50:
                    examples_df = pd.DataFrame(self.examples_data[regex_pattern])
                    self._save_examples_to_s3(rule_info, examples_df)

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
                        texts_before = text_before_pattern.split(_DELIM)
                        texts_after = joined_text.split(_DELIM)
                        
                        # Collect examples for this pattern
                        for i, (orig, cleaned) in enumerate(zip(texts_before, texts_after)):
                            if orig != cleaned:
                                self._collect_example(orig, cleaned, pattern.pattern, rule_info)
                                # Stop if we've collected enough examples for all patterns
                                if all(count >= 50 for count in self.examples_collected.values()):
                                    logger.info("Collected 50 examples for all patterns. Stopping debug collection.")
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
        
        # If in debug mode and we've collected examples for all patterns, stop processing
        if self.debug_mode and all(count >= 50 for count in self.examples_collected.values()):
            logger.info("Debug mode: Collected 50 examples for all patterns. Stopping processing.")
            return cleaned_df
            
        return cleaned_df

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
            'patterns_with_50_examples': sum(1 for count in self.examples_collected.values() if count >= 50)
        }