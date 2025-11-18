import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def check_stationarity(series):
    """
    Check if a time series is stationary using Augmented Dickey-Fuller test
    """
    try:
        result = adfuller(series.dropna())
        
        adf_stat = result[0]
        p_value = result[1]
        critical_values = result[4]
        
        is_stationary = p_value < 0.05
        
        return {
            'is_stationary': is_stationary,
            'adf_statistic': adf_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'interpretation': 'Stationary' if is_stationary else 'Non-stationary'
        }
    except Exception as e:
        return {
            'error': str(e),
            'is_stationary': False
        }

def decompose_time_series(series, period=12, model_type='additive'):
    """
    Decompose time series into trend, seasonal, and residual components
    """
    try:
        if len(series) < 2 * period:
            return {
                'error': f'Series length ({len(series)}) is too short for period {period}'
            }
        
        decomposition = seasonal_decompose(
            series, 
            model=model_type, 
            period=period,
            extrapolate_trend='freq'
        )
        
        return {
            'observed': decomposition.observed,
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'model_type': model_type,
            'period': period
        }
    except Exception as e:
        return {
            'error': str(e)
        }

def fit_arima_model(series, order=(1,1,1), forecast_steps=10):
    """
    Fit ARIMA model and make forecasts
    """
    try:
        # Fit ARIMA model
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        
        # Make forecasts
        forecast = fitted_model.forecast(steps=forecast_steps)
        forecast_ci = fitted_model.get_forecast(steps=forecast_steps).conf_int()
        
        # Get model statistics
        aic = fitted_model.aic
        bic = fitted_model.bic
        
        return {
            'model': fitted_model,
            'forecast': forecast,
            'forecast_ci': forecast_ci,
            'aic': aic,
            'bic': bic,
            'order': order,
            'summary': str(fitted_model.summary())
        }
    except Exception as e:
        return {
            'error': str(e)
        }

def fit_sarima_model(series, order=(1,1,1), seasonal_order=(1,1,1,12), forecast_steps=10):
    """
    Fit SARIMA model and make forecasts
    """
    try:
        # Fit SARIMA model
        model = ARIMA(series, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit()
        
        # Make forecasts
        forecast = fitted_model.forecast(steps=forecast_steps)
        forecast_ci = fitted_model.get_forecast(steps=forecast_steps).conf_int()
        
        # Get model statistics
        aic = fitted_model.aic
        bic = fitted_model.bic
        
        return {
            'model': fitted_model,
            'forecast': forecast,
            'forecast_ci': forecast_ci,
            'aic': aic,
            'bic': bic,
            'order': order,
            'seasonal_order': seasonal_order,
            'summary': str(fitted_model.summary())
        }
    except Exception as e:
        return {
            'error': str(e)
        }

def auto_arima(series, max_p=3, max_d=2, max_q=3):
    """
    Automatically find the best ARIMA parameters
    """
    try:
        best_aic = float('inf')
        best_order = None
        best_model = None
        
        # Grid search for best parameters
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                            best_model = fitted_model
                    except:
                        continue
        
        if best_model is None:
            return {
                'error': 'Could not fit any ARIMA model'
            }
        
        # Make forecast with best model
        forecast = best_model.forecast(steps=10)
        
        return {
            'best_order': best_order,
            'best_aic': best_aic,
            'best_bic': best_model.bic,
            'forecast': forecast,
            'model': best_model
        }
    except Exception as e:
        return {
            'error': str(e)
        }

def plot_acf_pacf(series, lags=40):
    """
    Plot ACF and PACF for time series analysis
    """
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # ACF plot
        plot_acf(series, lags=lags, ax=ax1)
        ax1.set_title('Autocorrelation Function (ACF)')
        
        # PACF plot
        plot_pacf(series, lags=lags, ax=ax2)
        ax2.set_title('Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        return {
            'error': str(e)
        }

def create_time_series_plot(series, title="Time Series Plot"):
    """
    Create an interactive time series plot
    """
    try:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines',
            name='Time Series',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    except Exception as e:
        return {
            'error': str(e)
        }

def create_forecast_plot(historical_data, forecast, forecast_ci=None, title="Time Series Forecast"):
    """
    Create a plot showing historical data and forecasts
    """
    try:
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data.values,
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Forecast
        forecast_index = pd.date_range(
            start=historical_data.index[-1],
            periods=len(forecast) + 1,
            freq='D'
        )[1:]
        
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Confidence intervals
        if forecast_ci is not None:
            fig.add_trace(go.Scatter(
                x=forecast_index,
                y=forecast_ci.iloc[:, 0],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_index,
                y=forecast_ci.iloc[:, 1],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                name='Confidence Interval'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    except Exception as e:
        return {
            'error': str(e)
        }

def calculate_seasonality_strength(series, period=12):
    """
    Calculate the strength of seasonality in a time series
    """
    try:
        decomposition = seasonal_decompose(series, period=period, extrapolate_trend='freq')
        
        # Calculate strength of seasonality
        seasonal_var = np.var(decomposition.seasonal)
        residual_var = np.var(decomposition.resid)
        
        if seasonal_var + residual_var == 0:
            return 0
        
        seasonality_strength = seasonal_var / (seasonal_var + residual_var)
        
        return seasonality_strength
    except Exception as e:
        return 0

def detect_outliers_ts(series, method='iqr', threshold=1.5):
    """
    Detect outliers in time series data
    """
    try:
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = (series < lower_bound) | (series > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = z_scores > threshold
        
        else:
            outliers = pd.Series(False, index=series.index)
        
        return {
            'outliers': outliers,
            'outlier_count': outliers.sum(),
            'outlier_percentage': (outliers.sum() / len(series)) * 100,
            'method': method,
            'threshold': threshold
        }
    except Exception as e:
        return {
            'error': str(e),
            'outliers': pd.Series(False, index=series.index)
        }