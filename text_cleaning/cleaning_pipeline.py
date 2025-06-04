# pipelines/cleaning_pipeline.py
import pandas as pd

class CleaningPipeline:
    def __init__(self, fetcher, cleaner, source_name: str):
        self.fetcher = fetcher
        self.cleaner = cleaner
        self.source_name = source_name

    def run(self):
        df = self.fetcher.fetch_raw_data()
        df = self.cleaner.clean(df)
        self.fetcher.save_cleaned_data(df, self.source_name)
        print(f"[✓] Cleaning pipeline completed.")