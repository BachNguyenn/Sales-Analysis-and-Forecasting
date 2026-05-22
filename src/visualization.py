import matplotlib.pyplot as plt


def plot_monthly_sales(monthly_sales, save_path=None):
    """
    Vẽ biểu đồ doanh số theo tháng.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sales["Date"], monthly_sales["Sales"], marker="o")
    plt.title("Monthly Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_predictions(date_test, y_test, lr_pred, rf_pred, save_path=None):
    """
    Vẽ biểu đồ so sánh doanh số thực tế và dự báo.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(date_test, y_test, label="Actual Sales", marker="o")
    plt.plot(date_test, lr_pred, label="Linear Regression", marker="o")
    plt.plot(date_test, rf_pred, label="Random Forest", marker="o")

    plt.title("Actual Sales vs Forecast Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_model_comparison(comparison, save_path=None):
    """
    Vẽ biểu đồ so sánh MAE và RMSE giữa các mô hình.
    """

    comparison.plot(
        x="Model",
        y=["MAE", "RMSE"],
        kind="bar",
        figsize=(8, 5)
    )

    plt.title("Model Performance Comparison")
    plt.ylabel("Error")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()