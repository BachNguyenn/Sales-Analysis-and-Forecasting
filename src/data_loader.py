import pandas as pd


def load_raw_data(file_path="data/raw/train.csv"):
    df = pd.read_csv(file_path)
    return df
    