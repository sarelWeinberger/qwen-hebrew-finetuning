from text_cleaning.fetchers.base_fetcher import BaseFetcher
import pandas as pd


class LocalSourceFetcher(BaseFetcher):
    def __init__(self, file_path: str, output_path: str):
        self.file_path = file_path
        self.output_path = output_path

    def fetch_raw_data(self) -> pd.DataFrame:
        """
        Fetch raw data from a local CSV file.
        """
        try:
            df = pd.read_csv(self.file_path, header=None, names=['text', 'n_count'])
            print(f"[✓] Successfully fetched data from {self.file_path}")

            return df
        except FileNotFoundError:
            print(f"[✗] File not found: {self.file_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"[✗] Error fetching data: {e}")
            return pd.DataFrame()


    def save_cleaned_data(self, df: pd.DataFrame):
        """
        Save cleaned data to a local CSV file.
        """
        try:
            df.to_csv(self.output_path, index=False)
            print(f"[✓] Saved cleaned data to {self.output_path}")
        except Exception as e:
            print(f"[✗] Error saving cleaned data: {e}")