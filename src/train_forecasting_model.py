"""
Sales Forecasting & Demand Analytics
Model Training & Evaluation Pipeline
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_data(filepath: str) -> pd.DataFrame:
    """Load sales dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath, encoding="utf-8")
    return df


def preprocess_data(df: pd.DataFrame):
    """
    Clean dataset, parse dates, sort chronologically, and aggregate sales.
    Returns:
        raw_clean_df: Cleaned transaction DataFrame with calendar features.
        daily_sales: Daily aggregated sales Series with continuous frequency.
    """
    df = df.copy()
    
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    
    # Parse Order Date to datetime
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df = df.sort_values('Order Date').reset_index(drop=True)
    
    # Handle missing values if any
    df = df.dropna(subset=['Order Date', 'Sales'])
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Create calendar and time features
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Month_Name'] = df['Order Date'].dt.strftime('%b')
    df['Week'] = df['Order Date'].dt.isocalendar().week
    df['Day_of_Week'] = df['Order Date'].dt.day_name()
    df['Day_of_Week_Num'] = df['Order Date'].dt.dayofweek
    df['Quarter'] = df['Order Date'].dt.quarter
    
    # Aggregate sales by date (fill missing dates with 0 sales for time-series continuity)
    daily_sales = (
        df.groupby('Order Date')['Sales']
        .sum()
        .asfreq('D', fill_value=0.0)
    )
    daily_sales.name = 'Sales'
    
    return df, daily_sales


def train_test_split_data(series: pd.Series, test_ratio: float = 0.2):
    """Chronological train/test split without shuffling."""
    split_idx = int(len(series) * (1 - test_ratio))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    return train, test


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series, model_name: str) -> dict:
    """Compute MAE, RMSE, and MAPE metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate MAPE for non-zero points to prevent division by zero
    non_zero = y_true > 0
    if np.sum(non_zero) > 0:
        mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    else:
        mape = np.nan
        
    return {
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2) if not np.isnan(mape) else None
    }


def train_and_evaluate(daily_sales: pd.Series):
    """
    Train ARIMA and Exponential Smoothing models on train split,
    evaluate on test split, and select the best model.
    """
    print(f"Total time series length: {len(daily_sales)} days")
    train, test = train_test_split_data(daily_sales, test_ratio=0.2)
    print(f"Training set: {len(train)} days ({train.index.min().date()} to {train.index.max().date()})")
    print(f"Testing set: {len(test)} days ({test.index.min().date()} to {test.index.max().date()})")
    
    # 1. ARIMA Model
    print("\n--- Training ARIMA(1, 1, 1) ---")
    arima_train = ARIMA(train, order=(1, 1, 1)).fit()
    arima_preds = arima_train.forecast(steps=len(test))
    arima_metrics = evaluate_forecast(test, arima_preds, "ARIMA(1,1,1)")
    print(f"ARIMA Results: MAE={arima_metrics['MAE']}, RMSE={arima_metrics['RMSE']}, MAPE={arima_metrics['MAPE (%)']}%")
    
    # 2. Holt-Winters Exponential Smoothing Model
    print("\n--- Training Holt-Winters Exponential Smoothing (Additive, S=7) ---")
    hw_train = ExponentialSmoothing(
        train,
        trend='add',
        seasonal='add',
        seasonal_periods=7,
        initialization_method='estimated'
    ).fit()
    hw_preds = hw_train.forecast(steps=len(test))
    hw_metrics = evaluate_forecast(test, hw_preds, "Holt-Winters Exp Smoothing")
    print(f"Holt-Winters Results: MAE={hw_metrics['MAE']}, RMSE={hw_metrics['RMSE']}, MAPE={hw_metrics['MAPE (%)']}%")
    
    metrics_df = pd.DataFrame([arima_metrics, hw_metrics])
    
    # Select best model based on MAE
    best_model_name = "Holt-Winters Exp Smoothing" if hw_metrics['MAE'] < arima_metrics['MAE'] else "ARIMA(1,1,1)"
    print(f"\nOptimal Model based on MAE: {best_model_name}")
    
    # Fit final models on full dataset for future forecasting
    print("\n--- Fitting final models on full historical dataset ---")
    final_arima = ARIMA(daily_sales, order=(1, 1, 1)).fit()
    final_hw = ExponentialSmoothing(
        daily_sales,
        trend='add',
        seasonal='add',
        seasonal_periods=7,
        initialization_method='estimated'
    ).fit()
    
    # Generate 30-day future forecast
    forecast_steps = 30
    last_date = daily_sales.index.max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps, freq='D')
    
    # ARIMA forecast with confidence intervals
    arima_full_fc = final_arima.get_forecast(steps=forecast_steps)
    arima_fc_mean = arima_full_fc.predicted_mean
    arima_ci = arima_full_fc.conf_int(alpha=0.05)
    
    # Holt-Winters forecast
    hw_fc_mean = final_hw.forecast(steps=forecast_steps)
    
    # Compute empirical standard error from residuals for Holt-Winters confidence intervals
    hw_resid_std = np.std(final_hw.resid)
    hw_lower = np.maximum(0, hw_fc_mean - 1.96 * hw_resid_std)
    hw_upper = hw_fc_mean + 1.96 * hw_resid_std
    
    # Primary forecast results dataframe
    forecast_results = pd.DataFrame({
        'Date': future_dates.strftime('%Y-%m-%d'),
        'Forecast_ARIMA': np.round(arima_fc_mean.values, 2),
        'ARIMA_Lower_CI': np.round(arima_ci.iloc[:, 0].values, 2),
        'ARIMA_Upper_CI': np.round(arima_ci.iloc[:, 1].values, 2),
        'Forecast_HoltWinters': np.round(hw_fc_mean.values, 2),
        'HoltWinters_Lower_CI': np.round(hw_lower.values, 2),
        'HoltWinters_Upper_CI': np.round(hw_upper.values, 2),
        'Primary_Forecast': np.round(hw_fc_mean.values if best_model_name.startswith("Holt") else arima_fc_mean.values, 2),
        'Primary_Lower_CI': np.round(hw_lower.values if best_model_name.startswith("Holt") else np.maximum(0, arima_ci.iloc[:, 0].values), 2),
        'Primary_Upper_CI': np.round(hw_upper.values if best_model_name.startswith("Holt") else arima_ci.iloc[:, 1].values, 2)
    })
    
    # Build complete model package bundle
    model_package = {
        'best_model_name': best_model_name,
        'arima_model': final_arima,
        'hw_model': final_hw,
        'metrics_df': metrics_df,
        'train_test_split': {
            'train_start': train.index.min().strftime('%Y-%m-%d'),
            'train_end': train.index.max().strftime('%Y-%m-%d'),
            'test_start': test.index.min().strftime('%Y-%m-%d'),
            'test_end': test.index.max().strftime('%Y-%m-%d'),
            'test_actuals': test,
            'arima_test_preds': pd.Series(arima_preds.values, index=test.index),
            'hw_test_preds': pd.Series(hw_preds.values, index=test.index)
        },
        'forecast_steps': forecast_steps,
        'last_historical_date': str(last_date.date())
    }
    
    return model_package, forecast_results, metrics_df


def save_artifacts(model_package: dict, forecast_results: pd.DataFrame, metrics_df: pd.DataFrame, output_dir: str):
    """Save model binary, forecast results, and evaluation metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save forecasting model
    model_path = os.path.join(output_dir, "forecasting_model.pkl")
    joblib.dump(model_package, model_path)
    print(f"Trained model package saved to: {model_path}")
    
    # 2. Save forecast results CSV
    results_path = os.path.join(output_dir, "forecast_results.csv")
    forecast_results.to_csv(results_path, index=False)
    print(f"Forecast results saved to: {results_path}")
    
    # 3. Save metrics summary CSV
    metrics_path = os.path.join(output_dir, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Model evaluation metrics saved to: {metrics_path}")


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(base_dir, "data", "sales_data.csv")
    models_dir = os.path.join(base_dir, "models")
    
    print("=" * 60)
    print("SALES FORECASTING & DEMAND ANALYTICS - TRAINING PIPELINE")
    print("=" * 60)
    print(f"Loading data from: {data_path}")
    
    raw_df = load_data(data_path)
    print(f"Loaded raw dataset with shape: {raw_df.shape}")
    
    raw_clean_df, daily_sales = preprocess_data(raw_df)
    print(f"Data cleaned and aggregated. Total days: {len(daily_sales)}")
    
    model_package, forecast_results, metrics_df = train_and_evaluate(daily_sales)
    
    save_artifacts(model_package, forecast_results, metrics_df, models_dir)
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print("\nNext 5 Days Future Sales Forecast Preview:")
    print(forecast_results[['Date', 'Primary_Forecast', 'Primary_Lower_CI', 'Primary_Upper_CI']].head().to_string(index=False))
    print("=" * 60)
    print("Training pipeline successfully finished!")


if __name__ == "__main__":
    main()
