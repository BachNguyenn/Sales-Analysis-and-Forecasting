import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_COLS = [
    "year",
    "month",
    "quarter",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "rolling_mean_6",
]


def split_data(data: pd.DataFrame, train_ratio: float = 0.8):
    """Split time-series data chronologically into train and test sets."""
    X = data[FEATURE_COLS]
    y = data["Sales"]

    train_size = int(len(data) * train_ratio)

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]
    date_test = data["Date"].iloc[train_size:]

    return X_train, X_test, y_train, y_test, date_test


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=2, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = mean_absolute_percentage_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    return pred, mae, rmse, mape, r2


def save_model(model, file_path: str):
    joblib.dump(model, file_path)


def get_best_model(comparison: pd.DataFrame, models: dict, metric: str = "RMSE"):
    """Return best model name and object by lowest error metric."""
    best_model_name = comparison.sort_values(metric).iloc[0]["Model"]
    return best_model_name, models[best_model_name]


def generate_future_forecast(last_data: pd.DataFrame, model, periods: int = 12) -> pd.DataFrame:
    """Generate recursive monthly forecast using the selected trained model."""
    future_predictions = []
    working = last_data.copy().reset_index(drop=True)

    for _ in range(periods):
        next_date = working["Date"].iloc[-1] + pd.DateOffset(months=1)

        new_row = {
            "Date": next_date,
            "year": next_date.year,
            "month": next_date.month,
            "quarter": next_date.quarter,
            "month_sin": np.sin(2 * np.pi * next_date.month / 12),
            "month_cos": np.cos(2 * np.pi * next_date.month / 12),
            "lag_1": working["Sales"].iloc[-1],
            "lag_2": working["Sales"].iloc[-2],
            "lag_3": working["Sales"].iloc[-3],
            "rolling_mean_3": working["Sales"].iloc[-3:].mean(),
            "rolling_mean_6": working["Sales"].iloc[-6:].mean(),
        }

        X_future = pd.DataFrame([new_row])[FEATURE_COLS]
        predicted_sales = model.predict(X_future)[0]
        new_row["Sales"] = predicted_sales

        future_predictions.append(new_row)
        working = pd.concat([working, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(future_predictions)
