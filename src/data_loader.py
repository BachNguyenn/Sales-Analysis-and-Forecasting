import pandas as pd


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Read raw sales dataset from CSV file."""
    return pd.read_csv(file_path)
