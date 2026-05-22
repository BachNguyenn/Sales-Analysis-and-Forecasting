import pandas as pd


def preprocess_sales_data(df):
    """
    Tiền xử lý dữ liệu bán hàng:
    - Chuyển Order Date sang datetime
    - Sắp xếp theo thời gian
    - Gom doanh số theo tháng
    """

    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df = df.sort_values("Order Date")

    monthly_sales = df.resample("M", on="Order Date")["Sales"].sum().reset_index()
    monthly_sales.columns = ["Date", "Sales"]

    return monthly_sales


def create_features(monthly_sales):
    """
    Tạo các biến đầu vào cho mô hình Machine Learning.
    """

    data = monthly_sales.copy()

    data["year"] = data["Date"].dt.year
    data["month"] = data["Date"].dt.month
    data["quarter"] = data["Date"].dt.quarter

    data["lag_1"] = data["Sales"].shift(1)
    data["lag_2"] = data["Sales"].shift(2)
    data["lag_3"] = data["Sales"].shift(3)

    data["rolling_mean_3"] = data["Sales"].shift(1).rolling(window=3).mean()
    data["rolling_mean_6"] = data["Sales"].shift(1).rolling(window=6).mean()

    data = data.dropna()

    return data