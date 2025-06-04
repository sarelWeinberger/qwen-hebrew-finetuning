import pandas as pd

class BaseCleaner:
    def clean(self, df: pd.DataFrame) -> str:
        raise NotImplementedError