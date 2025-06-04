import boto3
import pandas as pd
import io
from .base_fetcher import BaseFetcher

class S3SourceFetcher(BaseFetcher):
    def __init__(self, bucket_name: str, prefix: str, source_name: str, output_prefix: str):
        super().__init__(source_name)
        self.s3 = boto3.client("s3")
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/") + "/"
        self.output_prefix = output_prefix

    def fetch_raw_data(self) -> pd.DataFrame:
        paginator = self.s3.get_paginator('list_objects_v2')
        csv_keys = []

        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                # Fetch files that start with source_name and end with .csv
                filename = key.split('/')[-1]
                if filename.startswith(self.source_name) and filename.endswith('.csv'):
                    csv_keys.append(key)

        dataframes = []
        for key in csv_keys:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            df = pd.read_csv(io.BytesIO(response["Body"].read()))
            dataframes.append(df)

        return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

    def save_cleaned_data(self, df: pd.DataFrame):
        output_key = f"{self.output_prefix.rstrip('/')}/{self.source_name}_cleaned.csv"
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        self.s3.put_object(Bucket=self.bucket_name, Key=output_key, Body=csv_buffer.getvalue())
        print(f"[✓] Saved cleaned data to s3://{self.bucket_name}/{output_key}")