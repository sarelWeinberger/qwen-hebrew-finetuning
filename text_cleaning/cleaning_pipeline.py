# pipelines/cleaning_pipeline.py
import pandas as pd

class CleaningPipeline:
    def __init__(self, fetcher, cleaner, output_path):
        self.fetcher = fetcher
        self.cleaner = cleaner

    def run(self):
        df = self.fetcher.fetch_raw_data()
        df["clean_text"] = df["original_text"].apply(self.cleaner.clean)
        self.fetcher.save_cleaned_data(df)
        print(f"[✓] Cleaning pipeline completed.")