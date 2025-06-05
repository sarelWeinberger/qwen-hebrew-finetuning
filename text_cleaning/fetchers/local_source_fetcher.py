from fetchers.base_fetcher import BaseFetcher
import pandas as pd
from pathlib import Path
import glob
import time
import os
from ..logger import logger

class LocalSourceFetcher(BaseFetcher):
    def __init__(self, file_path: str, output_path: str):
        super().__init__(source_name=Path(file_path).stem)
        self.file_path = file_path
        self.output_path = output_path
        logger.info(f"Initialized LocalSourceFetcher with file path: {file_path}")
        logger.info(f"Output path: {output_path}")

    def get_files_to_process(self) -> list[str]:
        """
        Get all files matching the pattern in file_path.
        If file_path is a directory, get all CSV files in it.
        If file_path is a specific file, return just that file.
        """
        path = Path(self.file_path)
        if path.is_dir():
            files = list(glob.glob(str(path / "*.csv")))
            logger.info(f"Found {len(files)} CSV files in directory: {self.file_path}")
            return files
        elif path.is_file():
            logger.info(f"Processing single file: {self.file_path}")
            return [str(path)]
        else:
            # Handle glob patterns
            files = list(glob.glob(self.file_path))
            logger.info(f"Found {len(files)} files matching pattern: {self.file_path}")
            return files

    def fetch_single_file(self, file_path: str) -> pd.DataFrame:
        """
        Fetch raw data from a single local CSV file.
        """
        start_time = time.time()
        file_stats = {
            'rows': 0,
            'size_bytes': 0,
            'processing_time': 0
        }
        
        try:
            # Get file size
            file_stats['size_bytes'] = os.path.getsize(file_path)
            logger.info(f"Reading file: {file_path} (Size: {file_stats['size_bytes']} bytes)")
            
            df = pd.read_csv(file_path, header=None, names=['text', 'n_count'])
            file_stats['rows'] = len(df)
            
            # Update statistics
            self.stats['total_files_processed'] += 1
            self.stats['total_rows_fetched'] += len(df)
            self.stats['total_bytes_read'] += file_stats['size_bytes']
            
            logger.info(f"Successfully fetched {len(df)} rows from {file_path}")
            return df
            
        except FileNotFoundError:
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            return pd.DataFrame()
            
        except Exception as e:
            error_msg = f"Error fetching data from {file_path}: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            return pd.DataFrame()
            
        finally:
            file_stats['processing_time'] = time.time() - start_time
            self.stats['file_stats'][file_path] = file_stats

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
            
            # Save the data
            df.to_csv(output_path, index=False)
            
            # Log success
            logger.info(f"Saved {len(df)} rows to {output_path}")
            logger.info(f"Output file size: {os.path.getsize(output_path)} bytes")
            
        except Exception as e:
            error_msg = f"Error saving cleaned data to {output_path}: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)