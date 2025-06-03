import boto3
import pandas as pd
import io
from text_cleaning.fetchers.base_fetcher import BaseFetcher

class S3SourceFetcher(BaseFetcher):
    def __init__(self, bucket_name: str, prefix: str, source_name: str):
        super().__init__(source_name)
        self.s3 = boto3.client("s3")
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/") + "/"

    def fetch_raw_data(self) -> pd.DataFrame:
        # Get list of matching CSVs like source1_part1.csv, source1_part2.csv
        response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=self.prefix)
        if 'Contents' not in response:
            return pd.DataFrame()

        matching_keys = [
            obj['Key']
            for obj in response['Contents']
            if obj['Key'].endswith(".csv") and f"{self.source_name}_part" in obj['Key']
        ]

        # Load and concatenate
        dataframes = []
        for key in matching_keys:
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            df = pd.read_csv(io.BytesIO(obj["Body"].read()))
            dataframes.append(df)

        if dataframes:
            return pd.concat(dataframes, ignore_index=True)
        else:
            return pd.DataFrame()