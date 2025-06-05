from .base_cleaner import BaseCleaner
import pandas as pd
import time
from utils.logger import logger


class DuplicateRemoverCleaner(BaseCleaner):
    def __init__(self):
        super().__init__()
        logger.info("Initialized DuplicateRemoverCleaner")

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate lines from the 'text' column of the DataFrame.
        """
        start_time = time.time()
        res_df = pd.DataFrame(columns=['text', 'n_count'])
        new_texts = []
        new_n_count = []
        
        self.stats['total_rows_processed'] = len(df)
        total_lines_removed = 0
        total_original_lines = 0
        
        for idx, text in enumerate(df['text']):
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            total_original_lines += len(lines)
            seen = set()
            unique_lines = []
            
            for line in lines:
                if line not in seen:
                    seen.add(line)
                    unique_lines.append(line)
                else:
                    total_lines_removed += 1
            
            new_text = '\n'.join(unique_lines)
            new_n_count.append(len(new_text.split()))
            new_texts.append(new_text)
            
            if len(unique_lines) != len(lines):
                self.stats['rows_modified'] += 1

        res_df['text'] = new_texts
        res_df['n_count'] = new_n_count
        res_df = res_df.drop_duplicates()
        
        # Update statistics
        self.stats['execution_time'] = time.time() - start_time
        self.stats['patterns_matched'] = {
            'duplicate_lines': total_lines_removed,
            'total_original_lines': total_original_lines
        }
        
        # Log detailed statistics
        logger.info(f"Processed {len(df)} texts rows")
        logger.info(f"Modified {self.stats['rows_modified']} texts rows")
        logger.info(f"Total lines processed: {total_original_lines}")
        logger.info(f"Duplicate lines removed: {total_lines_removed}")
        logger.info(f"Duplicate removal rate: {(total_lines_removed/total_original_lines*100):.2f}%")
        logger.info(f"Execution time: {self.stats['execution_time']:.2f} seconds")
        
        return res_df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.clean_dataframe(df)
