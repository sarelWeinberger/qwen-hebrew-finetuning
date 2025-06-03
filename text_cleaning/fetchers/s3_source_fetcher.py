import boto3
import pandas as pd
import io
from .base_fetcher import BaseFetcher

class S3SourceFetcher(BaseFetcher):
    def __init__(self, bucket_name: str, prefix: str, source_name: str, output_prefix: str = None):
        super().__init__(source_name)
        self.s3 = boto3.client("s3")
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/") + "/"
        self.output_prefix = output_prefix or "cleaned-data/"  # default prefix

    def fetch_raw_data(self) -> pd.DataFrame:
        response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=self.prefix)
        if 'Contents' not in response:
            return pd.DataFrame()

        matching_keys = [
            obj['Key']
            for obj in response['Contents']
            if obj['Key'].endswith(".csv") and f"{self.source_name}_part" in obj['Key']
        ]

        dataframes = []
        for key in matching_keys:
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            df = pd.read_csv(io.BytesIO(obj["Body"].read()))
            dataframes.append(df)

        return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

    def save_cleaned_data(self, df: pd.DataFrame):
        output_key = f"{self.output_prefix}{self.source_name}_cleaned.csv"
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        self.s3.put_object(Bucket=self.bucket_name, Key=output_key, Body=csv_buffer.getvalue())
        print(f"[✓] Saved cleaned data to s3://{self.bucket_name}/{output_key}")