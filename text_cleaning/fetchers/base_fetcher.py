class BaseFetcher:
    def __init__(self, source_name: str):
        self.source_name = source_name

    def fetch_raw_data(self):
        raise NotImplementedError