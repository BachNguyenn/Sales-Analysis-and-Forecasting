import matplotlib.pyplot as plt
import pandas as pd


def _save_or_show(save_path=None):
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_monthly_sales(monthly_sales: pd.DataFrame, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sales["Date"], monthly_sales["Sales"], marker="o")
    plt.title("Monthly Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(True)
    _save_or_show(save_path)


def plot_bar_chart(data, title, xlabel, ylabel, save_path=None, figsize=(10, 5)):
    plt.figure(figsize=figsize)
    data.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.xticks(rotation=0)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_horizontal_bar_chart(data, title, xlabel, ylabel, save_path=None, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    data.sort_values().plot(kind="barh")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    _save_or_show(save_path)


def plot_predictions(date_test, y_test, predictions_dict, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(date_test, y_test, label="Actual Sales", marker="o", linewidth=2)

    for model_name, pred in predictions_dict.items():
        plt.plot(date_test, pred, label=model_name, marker="o")

    plt.title("Actual Sales vs Predicted Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True)
    _save_or_show(save_path)


def plot_model_comparison(comparison: pd.DataFrame, save_path=None):
    comparison.plot(x="Model", y=["MAE", "RMSE", "MAPE"], kind="bar", figsize=(10, 5))
    plt.title("Model Performance Comparison")
    plt.xlabel("Model")
    plt.ylabel("Error")
    plt.grid(True)
    plt.xticks(rotation=0)
    _save_or_show(save_path)


def plot_future_forecast(monthly_sales: pd.DataFrame, future_df: pd.DataFrame, best_model_name: str, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sales["Date"], monthly_sales["Sales"], label="Historical Sales")
    plt.plot(future_df["Date"], future_df["Sales"], label="Forecast Sales", marker="o")
    plt.title(f"Sales Forecast for Next 12 Months using {best_model_name}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True)
    _save_or_show(save_path)
