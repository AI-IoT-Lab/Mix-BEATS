from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np

def standardize_series(series, eps=1e-8):
    mean = np.mean(series)
    std = np.std(series)
    standardized_series = (series - mean) / (std + eps)
    return standardized_series, mean, std

def unscale_predictions(predictions, mean, std, eps=1e-8):
    return predictions * (std+eps) + mean



def decompose_series(series, period=24):
    """
    Decomposes a time series into trend and seasonal+residual components.
    Assumes hourly data by default (period=24).
    """
    result = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
    trend = result.trend
    seasonal_plus_resid = series - trend

    # Handle NaNs from the trend's boundary effects
    # trend = pd.Series(trend).fillna(method='bfill').fillna(method='ffill').values
    trend = pd.Series(trend).bfill().ffill().values
    seasonal_plus_resid = pd.Series(seasonal_plus_resid).fillna(0).values

    return trend, seasonal_plus_resid