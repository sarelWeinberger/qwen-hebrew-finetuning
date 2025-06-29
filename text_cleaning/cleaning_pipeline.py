# pipelines/cleaning_pipeline.py
import pandas as pd
from pathlib import Path
from utils.logger import logger

class CleaningPipeline:
    def __init__(self, fetcher, cleaner, source_name: str):
        self.fetcher = fetcher
        self.cleaner = cleaner
        self.source_name = source_name

    def run(self):
        # Get list of files to process
        files = self.fetcher.get_files_to_process()
        total_files = len(files)
        
        logger.info(f"Starting to process {total_files} files...")
        
        for i, file_path in enumerate(files, 1):
            logger.info(f"Processing file {i}/{total_files}: {file_path}")
            
            # Process single file
            df = self.fetcher.fetch_single_file(file_path)
            if not df.empty:
                # Extract file name for sample naming
                file_name = Path(file_path).stem
                
                # Clean the data (this will handle sample saving if enabled)
                df = self.cleaner.clean(df, file_name=file_name)
                
                # Save cleaned data
                self.fetcher.save_cleaned_data(df, self.source_name, file_path)
                logger.info(f"Completed processing file {i}/{total_files}")
            else:
                logger.warning(f"Skipped empty file {i}/{total_files}")
        
        logger.info(f"Cleaning pipeline completed for all {total_files} files.")