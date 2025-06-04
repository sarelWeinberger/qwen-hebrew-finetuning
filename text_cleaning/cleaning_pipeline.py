# pipelines/cleaning_pipeline.py
import pandas as pd
from pathlib import Path

class CleaningPipeline:
    def __init__(self, fetcher, cleaner, source_name: str):
        self.fetcher = fetcher
        self.cleaner = cleaner
        self.source_name = source_name

    def run(self):
        # Get list of files to process
        files = self.fetcher.get_files_to_process()
        total_files = len(files)
        
        print(f"[*] Starting to process {total_files} files...")
        
        for i, file_path in enumerate(files, 1):
            print(f"[*] Processing file {i}/{total_files}: {file_path}")
            
            # Process single file
            df = self.fetcher.fetch_single_file(file_path)
            if not df.empty:
                df = self.cleaner.clean(df)
                self.fetcher.save_cleaned_data(df, self.source_name, file_path)
                print(f"[✓] Completed processing file {i}/{total_files}")
            else:
                print(f"[!] Skipped empty file {i}/{total_files}")
        
        print(f"[✓] Cleaning pipeline completed for all {total_files} files.")