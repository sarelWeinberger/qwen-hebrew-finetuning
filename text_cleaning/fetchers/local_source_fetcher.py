from fetchers.base_fetcher import BaseFetcher
import pandas as pd
from pathlib import Path
import glob

class LocalSourceFetcher(BaseFetcher):
    def __init__(self, file_path: str, output_path: str):
        super().__init__(source_name=Path(file_path).stem)
        self.file_path = file_path
        self.output_path = output_path

    def get_files_to_process(self) -> list[str]:
        """
        Get all files matching the pattern in file_path.
        If file_path is a directory, get all CSV files in it.
        If file_path is a specific file, return just that file.
        """
        path = Path(self.file_path)
        if path.is_dir():
            return list(glob.glob(str(path / "*.csv")))
        elif path.is_file():
            return [str(path)]
        else:
            # Handle glob patterns
            return list(glob.glob(self.file_path))

    def fetch_single_file(self, file_path: str) -> pd.DataFrame:
        """
        Fetch raw data from a single local CSV file.
        """
        try:
            df = pd.read_csv(file_path, header=None, names=['text', 'n_count'])
            print(f"[✓] Successfully fetched data from {file_path}")
            return df
        except FileNotFoundError:
            print(f"[✗] File not found: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"[✗] Error fetching data: {e}")
            return pd.DataFrame()

    def save_cleaned_data(self, df: pd.DataFrame, source_name: str, original_file_path: str):
        """
        Save cleaned data to a local CSV file, preserving the original filename structure.
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(self.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get the original filename and create the output filename
            original_filename = Path(original_file_path).stem
            output_filename = f"{original_filename}_cleaned.csv"
            output_path = output_dir / output_filename
            
            df.to_csv(output_path, index=False)
            print(f"[✓] Saved cleaned data to {output_path}")
        except Exception as e:
            print(f"[✗] Error saving cleaned data: {e}")