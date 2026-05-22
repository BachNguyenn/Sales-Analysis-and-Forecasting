import pandas as pd


def clean_raw_data(df):
    """
    Làm sạch dữ liệu gốc.
    """

    df = df.copy()

    # Xóa dữ liệu trùng
    df = df.drop_duplicates()

    # Chuyển cột ngày sang datetime
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")

    # Xóa dòng lỗi ngày
    df = df.dropna(subset=["Order Date", "Ship Date"])

    # Xử lý giá trị thiếu ở Postal Code
    df["Postal Code"] = df["Postal Code"].fillna("Unknown")

    # Đảm bảo Sales là kiểu số
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

    # Xóa dòng thiếu Sales
    df = df.dropna(subset=["Sales"])

    # Chỉ giữ Sales > 0
    df = df[df["Sales"] > 0]

    # Sắp xếp theo thời gian
    df = df.sort_values("Order Date")

    return df


def create_monthly_sales(df):
    """
    Tổng hợp doanh số theo tháng.
    """

    monthly_sales = df.resample("ME", on="Order Date")["Sales"].sum().reset_index()
    monthly_sales.columns = ["Date", "Sales"]

    return monthly_sales


def create_features(monthly_sales):
    """
    Tạo đặc trưng cho mô hình Machine Learning.
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