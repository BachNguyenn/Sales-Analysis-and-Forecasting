import numpy as np
import pandas as pd


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Superstore sales data.

    Steps:
    - Remove duplicate rows.
    - Convert Order Date and Ship Date to datetime.
    - Fill missing Postal Code values.
    - Convert Sales to numeric.
    - Remove invalid dates and invalid sales values.
    - Create Delivery Days feature.
    - Sort data by Order Date.
    """
    data = df.copy()
    data = data.drop_duplicates()

    data["Order Date"] = pd.to_datetime(data["Order Date"], dayfirst=True, errors="coerce")
    data["Ship Date"] = pd.to_datetime(data["Ship Date"], dayfirst=True, errors="coerce")

    data = data.dropna(subset=["Order Date", "Ship Date"])

    if "Postal Code" in data.columns:
        data["Postal Code"] = data["Postal Code"].fillna("Unknown")

    data["Sales"] = pd.to_numeric(data["Sales"], errors="coerce")
    data = data.dropna(subset=["Sales"])
    data = data[data["Sales"] > 0]

    data["Delivery Days"] = (data["Ship Date"] - data["Order Date"]).dt.days
    data = data[data["Delivery Days"] >= 0]

    data = data.sort_values("Order Date").reset_index(drop=True)
    return data


def create_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sales by month for forecasting."""
    monthly_sales = df.resample("ME", on="Order Date")["Sales"].sum().reset_index()
    monthly_sales.columns = ["Date", "Sales"]
    return monthly_sales


def create_model_features(monthly_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Convert monthly sales time series into supervised-learning features.

    Features:
    - year, month, quarter
    - cyclical month encoding: month_sin, month_cos
    - lag features: lag_1, lag_2, lag_3
    - rolling means: rolling_mean_3, rolling_mean_6
    """
    data = monthly_sales.copy()
    data["Date"] = pd.to_datetime(data["Date"])

    data["year"] = data["Date"].dt.year
    data["month"] = data["Date"].dt.month
    data["quarter"] = data["Date"].dt.quarter

    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

    data["lag_1"] = data["Sales"].shift(1)
    data["lag_2"] = data["Sales"].shift(2)
    data["lag_3"] = data["Sales"].shift(3)

    data["rolling_mean_3"] = data["Sales"].shift(1).rolling(window=3).mean()
    data["rolling_mean_6"] = data["Sales"].shift(1).rolling(window=6).mean()

    data = data.dropna().reset_index(drop=True)
    return data


# Backward-compatible alias for older notebooks
create_features = create_model_features
