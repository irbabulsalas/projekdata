import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class AdvancedTimeSeriesModels:
    """
    Advanced Time Series Models for Data Analysis
    """
    
    @staticmethod
    def create_time_series_dashboard(df):
        """
        Create comprehensive time series dashboard
        """
        st.subheader("📈 Advanced Time Series Analysis")
        
        # Select time series column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ No numeric columns available for time series analysis")
            return
        
        # Data preparation
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Prep", "🧠 LSTM", "📈 Prophet", "🔍 Anomaly Detection", "📊 Advanced Stats"
        ])
        
        with tab1:
            AdvancedTimeSeriesModels.data_preparation(df, numeric_cols)
        
        with tab2:
            AdvancedTimeSeriesModels.lstm_model(df, numeric_cols)
        
        with tab3:
            AdvancedTimeSeriesModels.prophet_model(df, numeric_cols)
        
        with tab4:
            AdvancedTimeSeriesModels.anomaly_detection(df, numeric_cols)
        
        with tab5:
            AdvancedTimeSeriesModels.advanced_statistics(df, numeric_cols)
    
    @staticmethod
    def data_preparation(df, numeric_cols):
        """
        Data preparation for time series analysis
        """
        st.subheader("📊 Data Preparation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_col = st.selectbox("Select Time Series Column", numeric_cols)
            
            # Display basic statistics
            st.write("**Basic Statistics**")
            st.write(f"Mean: {df[selected_col].mean():.4f}")
            st.write(f"Std Dev: {df[selected_col].std():.4f}")
            st.write(f"Min: {df[selected_col].min():.4f}")
            st.write(f"Max: {df[selected_col].max():.4f}")
        
        with col2:
            # Time series plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[selected_col],
                mode='lines',
                name='Time Series',
                line=dict(color='blue')
            ))
            fig.update_layout(
                title=f'Time Series: {selected_col}',
                xaxis_title='Index',
                yaxis_title='Value'
            )
            st.plotly_chart(fig, use_container_width=True, key="ts_data_prep_original")
        
        # Data preprocessing options
        st.markdown("---")
        st.subheader("⚙️ Preprocessing Options")
        
        handle_missing = st.selectbox(
            "Handle Missing Values",
            ["drop", "forward_fill", "backward_fill", "interpolate"],
            help="Strategy for handling missing values"
        )
        
        scale_data = st.checkbox("Scale Data", value=True)
        
        if st.button("🔄 Apply Preprocessing"):
            with st.spinner("Preprocessing data..."):
                # Create a copy
                processed_df = df[[selected_col]].copy()
                
                # Handle missing values
                if handle_missing == "drop":
                    processed_df = processed_df.dropna()
                elif handle_missing == "forward_fill":
                    processed_df = processed_df.ffill()
                elif handle_missing == "backward_fill":
                    processed_df = processed_df.bfill()
                elif handle_missing == "interpolate":
                    processed_df = processed_df.interpolate()
                
                # Scale data
                if scale_data:
                    scaler = MinMaxScaler()
                    processed_df[selected_col] = scaler.fit_transform(processed_df[[selected_col]])
                    st.session_state['ts_scaler'] = scaler
                
                st.session_state['ts_processed_data'] = processed_df
                st.session_state['ts_selected_col'] = selected_col
                
                st.success(f"✅ Data preprocessed! Shape: {processed_df.shape}")
                
                # Display processed data
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=processed_df.index,
                    y=processed_df[selected_col],
                    mode='lines',
                    name='Processed Time Series',
                    line=dict(color='green')
                ))
                fig.update_layout(
                    title=f'Processed Time Series: {selected_col}',
                    xaxis_title='Index',
                    yaxis_title='Value'
                )
                st.plotly_chart(fig, use_container_width=True, key="ts_data_prep_processed")
    
    @staticmethod
    def lstm_model(df, numeric_cols):
        """
        LSTM Neural Network for time series forecasting
        """
        st.subheader("🧠 LSTM Time Series Model")
        
        # Check if data is preprocessed
        if 'ts_processed_data' not in st.session_state:
            st.warning("⚠️ Please preprocess data first in Data Prep tab")
            return
        
        processed_df = st.session_state['ts_processed_data']
        selected_col = st.session_state['ts_selected_col']
        
        # LSTM parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏗️ Model Architecture")
            
            sequence_length = st.slider("Sequence Length", 10, 100, 30)
            lstm_units = st.slider("LSTM Units", 10, 200, 50)
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
            
            num_layers = st.slider("Number of LSTM Layers", 1, 3, 1)
        
        with col2:
            st.subheader("⚙️ Training Parameters")
            
            epochs = st.slider("Epochs", 10, 200, 50)
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
            learning_rate = st.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=0)
            
            train_split = st.slider("Training Split", 0.6, 0.9, 0.8)
        
        if st.button("🚀 Train LSTM Model", type="primary"):
            with st.spinner("Training LSTM model..."):
                try:
                    # Import TensorFlow here to avoid import issues
                    import tensorflow as tf
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import LSTM, Dense, Dropout
                    from tensorflow.keras.optimizers import Adam
                    
                    # Prepare sequences
                    data = processed_df[selected_col].values
                    X, y = [], []
                    
                    for i in range(sequence_length, len(data)):
                        X.append(data[i-sequence_length:i])
                        y.append(data[i])
                    
                    X, y = np.array(X), np.array(y)
                    
                    # Reshape for LSTM
                    X = X.reshape((X.shape[0], X.shape[1], 1))
                    
                    # Split data
                    split_idx = int(len(X) * train_split)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    # Build LSTM model
                    model = Sequential()
                    
                    # Add LSTM layers
                    for i in range(num_layers):
                        return_sequences = i < num_layers - 1
                        model.add(LSTM(
                            lstm_units,
                            return_sequences=return_sequences,
                            input_shape=(sequence_length, 1) if i == 0 else None
                        ))
                        model.add(Dropout(dropout_rate))
                    
                    # Add output layer
                    model.add(Dense(1))
                    
                    # Compile model
                    model.compile(
                        optimizer=Adam(learning_rate=learning_rate),
                        loss='mse',
                        metrics=['mae']
                    )
                    
                    # Train model
                    history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        verbose=0
                    )
                    
                    # Make predictions
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    train_mse = mean_squared_error(y_train, train_pred)
                    test_mse = mean_squared_error(y_test, test_pred)
                    train_mae = mean_absolute_error(y_train, train_pred)
                    test_mae = mean_absolute_error(y_test, test_pred)
                    
                    # Store results
                    st.session_state['lstm_model'] = model
                    st.session_state['lstm_history'] = history
                    st.session_state['lstm_results'] = {
                        'train_mse': train_mse,
                        'test_mse': test_mse,
                        'train_mae': train_mae,
                        'test_mae': test_mae,
                        'y_test': y_test,
                        'test_pred': test_pred.flatten()
                    }
                    
                    st.success("✅ LSTM model trained successfully!")
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Train MSE", f"{train_mse:.6f}")
                        st.metric("Train MAE", f"{train_mae:.6f}")
                    with col2:
                        st.metric("Test MSE", f"{test_mse:.6f}")
                        st.metric("Test MAE", f"{test_mae:.6f}")
                    
                    # Plot training history
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("Training Loss", "Training MAE")
                    )
                    
                    fig.add_trace(
                        go.Scatter(y=history.history['loss'], name='Train Loss'),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(y=history.history['val_loss'], name='Val Loss'),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(y=history.history['mae'], name='Train MAE'),
                        row=1, col=2
                    )
                    fig.add_trace(
                        go.Scatter(y=history.history['val_mae'], name='Val MAE'),
                        row=1, col=2
                    )
                    
                    fig.update_layout(height=400, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True, key="lstm_training_history")
                    
                    # Plot predictions
                    fig = go.Figure()
                    
                    # Actual values
                    fig.add_trace(go.Scatter(
                        y=y_test,
                        mode='lines',
                        name='Actual',
                        line=dict(color='blue')
                    ))
                    
                    # Predicted values
                    fig.add_trace(go.Scatter(
                        y=test_pred.flatten(),
                        mode='lines',
                        name='Predicted',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title='LSTM Predictions vs Actual',
                        xaxis_title='Time',
                        yaxis_title='Value'
                    )
                    st.plotly_chart(fig, use_container_width=True, key="lstm_predictions")
                    
                except ImportError:
                    st.error("❌ TensorFlow is not installed. Please install it with: pip install tensorflow")
                except Exception as e:
                    st.error(f"❌ Error training LSTM model: {str(e)}")
    
    @staticmethod
    def prophet_model(df, numeric_cols):
        """
        Prophet model for time series forecasting
        """
        st.subheader("📈 Prophet Time Series Model")
        
        # Check if data is preprocessed
        if 'ts_processed_data' not in st.session_state:
            st.warning("⚠️ Please preprocess data first in Data Prep tab")
            return
        
        processed_df = st.session_state['ts_processed_data']
        selected_col = st.session_state['ts_selected_col']
        
        # Prophet parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Date Configuration")
            
            date_col = st.selectbox(
                "Date Column (optional)",
                [None] + df.columns.tolist(),
                help="Select a date column or use index"
            )
            
            if date_col:
                freq = st.selectbox(
                    "Frequency",
                    ["D", "W", "M", "Q", "Y"],
                    help="Data frequency: Daily, Weekly, Monthly, Quarterly, Yearly"
                )
            else:
                freq = 'D'
        
        with col2:
            st.subheader("⚙️ Model Parameters")
            
            yearly_seasonality = st.checkbox("Yearly Seasonality", value=True)
            weekly_seasonality = st.checkbox("Weekly Seasonality", value=True)
            daily_seasonality = st.checkbox("Daily Seasonality", value=False)
            
            changepoint_prior_scale = st.slider(
                "Changepoint Prior Scale",
                0.01, 0.5, 0.05,
                help="Flexibility of the trend"
            )
        
        if st.button("🚀 Train Prophet Model", type="primary"):
            with st.spinner("Training Prophet model..."):
                try:
                    from prophet import Prophet
                    
                    # Prepare data for Prophet
                    if date_col:
                        prophet_df = df[[date_col, selected_col]].copy()
                        prophet_df.columns = ['ds', 'y']
                    else:
                        prophet_df = pd.DataFrame({
                            'ds': pd.date_range(start='2020-01-01', periods=len(processed_df), freq=freq),
                            'y': processed_df[selected_col].values
                        })
                    
                    # Handle missing values
                    prophet_df = prophet_df.dropna()
                    
                    # Create and fit Prophet model
                    model = Prophet(
                        yearly_seasonality=yearly_seasonality,
                        weekly_seasonality=weekly_seasonality,
                        daily_seasonality=daily_seasonality,
                        changepoint_prior_scale=changepoint_prior_scale
                    )
                    
                    model.fit(prophet_df)
                    
                    # Make future dataframe and predictions
                    future = model.make_future_dataframe(periods=30, freq=freq)
                    forecast = model.predict(future)
                    
                    # Store results
                    st.session_state['prophet_model'] = model
                    st.session_state['prophet_forecast'] = forecast
                    
                    st.success("✅ Prophet model trained successfully!")
                    
                    # Plot forecast
                    fig1 = model.plot(forecast)
                    st.pyplot(fig1)
                    
                    # Plot components
                    fig2 = model.plot_components(forecast)
                    st.pyplot(fig2)
                    
                    # Display forecast table
                    st.subheader("📊 Forecast Table")
                    forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
                    forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                    st.dataframe(forecast_display, use_container_width=True)
                    
                except ImportError:
                    st.error("❌ Prophet is not installed. Please install it with: pip install prophet")
                except Exception as e:
                    st.error(f"❌ Error training Prophet model: {str(e)}")
    
    @staticmethod
    def anomaly_detection(df, numeric_cols):
        """
        Anomaly detection in time series
        """
        st.subheader("🔍 Anomaly Detection")
        
        # Select column
        selected_col = st.selectbox("Select Column for Anomaly Detection", numeric_cols)
        
        # Anomaly detection method
        method = st.selectbox(
            "Detection Method",
            ["statistical", "isolation_forest", "moving_average"],
            help="Method for detecting anomalies"
        )
        
        if method == "statistical":
            st.subheader("📊 Statistical Method")
            
            threshold = st.slider("Z-Score Threshold", 1.0, 4.0, 2.5)
            
            if st.button("🔍 Detect Anomalies"):
                with st.spinner("Detecting anomalies..."):
                    # Calculate Z-scores
                    z_scores = np.abs((df[selected_col] - df[selected_col].mean()) / df[selected_col].std())
                    anomalies = z_scores > threshold
                    
                    # Plot results
                    fig = go.Figure()
                    
                    # Normal points
                    normal_mask = ~anomalies
                    fig.add_trace(go.Scatter(
                        x=df.index[normal_mask],
                        y=df[selected_col][normal_mask],
                        mode='markers',
                        name='Normal',
                        marker=dict(color='blue', size=4)
                    ))
                    
                    # Anomalies
                    anomaly_mask = anomalies
                    fig.add_trace(go.Scatter(
                        x=df.index[anomaly_mask],
                        y=df[selected_col][anomaly_mask],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', size=8, symbol='x')
                    ))
                    
                    fig.update_layout(
                        title=f'Anomaly Detection: {selected_col}',
                        xaxis_title='Index',
                        yaxis_title='Value'
                    )
                    st.plotly_chart(fig, use_container_width=True, key="anomaly_statistical")
                    
                    # Display statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Points", len(df))
                    with col2:
                        st.metric("Anomalies", anomalies.sum())
                    with col3:
                        st.metric("Anomaly Rate", f"{(anomalies.sum()/len(df)*100):.2f}%")
        
        elif method == "isolation_forest":
            st.subheader("🌲 Isolation Forest Method")
            
            contamination = st.slider("Contamination Rate", 0.01, 0.2, 0.05)
            
            if st.button("🔍 Detect Anomalies"):
                with st.spinner("Detecting anomalies..."):
                    try:
                        from sklearn.ensemble import IsolationForest
                        
                        # Prepare data
                        X = df[[selected_col]].values
                        
                        # Fit Isolation Forest
                        iso_forest = IsolationForest(contamination=contamination, random_state=42)
                        anomaly_labels = iso_forest.fit_predict(X)
                        
                        # Convert to boolean (1 = normal, -1 = anomaly)
                        anomalies = anomaly_labels == -1
                        
                        # Plot results
                        fig = go.Figure()
                        
                        # Normal points
                        normal_mask = ~anomalies
                        fig.add_trace(go.Scatter(
                            x=df.index[normal_mask],
                            y=df[selected_col][normal_mask],
                            mode='markers',
                            name='Normal',
                            marker=dict(color='blue', size=4)
                        ))
                        
                        # Anomalies
                        anomaly_mask = anomalies
                        fig.add_trace(go.Scatter(
                            x=df.index[anomaly_mask],
                            y=df[selected_col][anomaly_mask],
                            mode='markers',
                            name='Anomalies',
                            marker=dict(color='red', size=8, symbol='x')
                        ))
                        
                        fig.update_layout(
                            title=f'Isolation Forest Anomaly Detection: {selected_col}',
                            xaxis_title='Index',
                            yaxis_title='Value'
                        )
                        st.plotly_chart(fig, use_container_width=True, key="anomaly_isolation_forest")
                        
                        # Display statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Points", len(df))
                        with col2:
                            st.metric("Anomalies", anomalies.sum())
                        with col3:
                            st.metric("Anomaly Rate", f"{(anomalies.sum()/len(df)*100):.2f}%")
                        
                    except Exception as e:
                        st.error(f"❌ Error with Isolation Forest: {str(e)}")
        
        elif method == "moving_average":
            st.subheader("📈 Moving Average Method")
            
            window_size = st.slider("Window Size", 5, 50, 20)
            threshold_factor = st.slider("Threshold Factor", 1.0, 5.0, 2.0)
            
            if st.button("🔍 Detect Anomalies"):
                with st.spinner("Detecting anomalies..."):
                    # Calculate moving average and standard deviation
                    moving_avg = df[selected_col].rolling(window=window_size).mean()
                    moving_std = df[selected_col].rolling(window=window_size).std()
                    
                    # Define thresholds
                    upper_threshold = moving_avg + threshold_factor * moving_std
                    lower_threshold = moving_avg - threshold_factor * moving_std
                    
                    # Detect anomalies
                    anomalies = (df[selected_col] > upper_threshold) | (df[selected_col] < lower_threshold)
                    
                    # Plot results
                    fig = go.Figure()
                    
                    # Time series
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[selected_col],
                        mode='lines',
                        name='Time Series',
                        line=dict(color='blue')
                    ))
                    
                    # Moving average
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=moving_avg,
                        mode='lines',
                        name='Moving Average',
                        line=dict(color='green', dash='dash')
                    ))
                    
                    # Thresholds
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=upper_threshold,
                        mode='lines',
                        name='Upper Threshold',
                        line=dict(color='red', dash='dot')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=lower_threshold,
                        mode='lines',
                        name='Lower Threshold',
                        line=dict(color='red', dash='dot'),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.1)'
                    ))
                    
                    # Anomalies
                    anomaly_mask = anomalies
                    if anomaly_mask.any():
                        fig.add_trace(go.Scatter(
                            x=df.index[anomaly_mask],
                            y=df[selected_col][anomaly_mask],
                            mode='markers',
                            name='Anomalies',
                            marker=dict(color='red', size=8, symbol='x')
                        ))
                    
                    fig.update_layout(
                        title=f'Moving Average Anomaly Detection: {selected_col}',
                        xaxis_title='Index',
                        yaxis_title='Value'
                    )
                    st.plotly_chart(fig, use_container_width=True, key="anomaly_moving_average")
                    
                    # Display statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Points", len(df))
                    with col2:
                        st.metric("Anomalies", anomalies.sum())
                    with col3:
                        st.metric("Anomaly Rate", f"{(anomalies.sum()/len(df)*100):.2f}%")
    
    @staticmethod
    def advanced_statistics(df, numeric_cols):
        """
        Advanced time series statistics
        """
        st.subheader("📊 Advanced Time Series Statistics")
        
        selected_col = st.selectbox("Select Column for Analysis", numeric_cols)
        
        # Calculate various statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Basic Statistics")
            
            series = df[selected_col]
            
            st.metric("Mean", f"{series.mean():.4f}")
            st.metric("Median", f"{series.median():.4f}")
            st.metric("Std Dev", f"{series.std():.4f}")
            st.metric("Variance", f"{series.var():.4f}")
            
            st.metric("Skewness", f"{series.skew():.4f}")
            st.metric("Kurtosis", f"{series.kurtosis():.4f}")
            st.metric("Range", f"{series.max() - series.min():.4f}")
            st.metric("CV", f"{series.std()/series.mean():.4f}")
        
        with col2:
            st.subheader("📊 Distribution Analysis")
            
            # Histogram
            fig = px.histogram(
                df, x=selected_col,
                title=f'Distribution of {selected_col}',
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True, key="advanced_stats_histogram")
            
            # Box plot
            fig = px.box(
                df, y=selected_col,
                title=f'Box Plot of {selected_col}'
            )
            st.plotly_chart(fig, use_container_width=True, key="advanced_stats_box")
        
        # Autocorrelation analysis
        st.markdown("---")
        st.subheader("🔗 Autocorrelation Analysis")
        
        try:
            from statsmodels.tsa.stattools import acf, pacf
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            
            # Calculate ACF and PACF
            max_lags = min(40, len(series) // 4)
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # ACF plot
            plot_acf(series, lags=max_lags, ax=ax1)
            ax1.set_title('Autocorrelation Function (ACF)')
            
            # PACF plot
            plot_pacf(series, lags=max_lags, ax=ax2)
            ax2.set_title('Partial Autocorrelation Function (PACF)')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Calculate and display key statistics
            acf_values = acf(series, nlags=max_lags)
            pacf_values = pacf(series, nlags=max_lags)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ACF Lag 1", f"{acf_values[1]:.4f}")
                st.metric("ACF Lag 5", f"{acf_values[5] if len(acf_values) > 5 else 0:.4f}")
            with col2:
                st.metric("PACF Lag 1", f"{pacf_values[1]:.4f}")
                st.metric("PACF Lag 5", f"{pacf_values[5] if len(pacf_values) > 5 else 0:.4f}")
            
        except Exception as e:
            st.warning(f"⚠️ Could not calculate autocorrelation: {str(e)}")
        
        # Trend analysis
        st.markdown("---")
        st.subheader("📈 Trend Analysis")
        
        # Simple linear trend
        x = np.arange(len(series))
        trend_coef = np.polyfit(x, series, 1)
        trend_line = np.polyval(trend_coef, x)
        
        # Calculate trend strength
        correlation = np.corrcoef(series, trend_line)[0, 1]
        trend_strength = correlation ** 2
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Trend Slope", f"{trend_coef[0]:.6f}")
            st.metric("Trend Strength", f"{trend_strength:.4f}")
        with col2:
            st.metric("Trend Direction", "Increasing" if trend_coef[0] > 0 else "Decreasing")
            st.metric("Trend Significance", "Strong" if trend_strength > 0.5 else "Weak" if trend_strength < 0.2 else "Moderate")
        
        # Plot with trend
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=series,
            mode='lines',
            name='Time Series',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=trend_line,
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'Trend Analysis: {selected_col}',
            xaxis_title='Index',
            yaxis_title='Value'
        )
        st.plotly_chart(fig, use_container_width=True, key="advanced_stats_trend")

# Create instance for import compatibility
advanced_timeseries = AdvancedTimeSeriesModels()