import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class DeepLearningModels:
    """
    Deep Learning Models for Data Analysis
    """
    
    @staticmethod
    def create_deep_learning_dashboard(df, target_column):
        """
        Create comprehensive deep learning dashboard
        """
        st.subheader("🧠 Deep Learning Models")
        
        # Data preparation
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Prep", "🧠 Neural Network", "📈 Training History", "🎯 Model Evaluation"])
        
        with tab1:
            DeepLearningModels.data_preparation(df, target_column)
        
        with tab2:
            DeepLearningModels.build_neural_network(df, target_column)
        
        with tab3:
            DeepLearningModels.training_history()
        
        with tab4:
            DeepLearningModels.model_evaluation(df, target_column)
    
    @staticmethod
    def data_preparation(df, target_column):
        """
        Data preparation for deep learning
        """
        st.subheader("📊 Data Preparation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Info**")
            st.write(f"Shape: {df.shape}")
            st.write(f"Target Column: {target_column}")
            
            # Check data types
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if target_column in categorical_cols:
                categorical_cols.remove(target_column)
            elif target_column in numeric_cols:
                numeric_cols.remove(target_column)
            
            st.write(f"Numeric Features: {len(numeric_cols)}")
            st.write(f"Categorical Features: {len(categorical_cols)}")
        
        with col2:
            st.write("**Target Distribution**")
            if df[target_column].dtype == 'object':
                fig = px.bar(df[target_column].value_counts(), 
                            title=f"Distribution of {target_column}")
                st.plotly_chart(fig, use_container_width=True, key="dl_target_dist_categorical")
            else:
                fig = px.histogram(df, x=target_column, 
                                 title=f"Distribution of {target_column}")
                st.plotly_chart(fig, use_container_width=True, key="dl_target_dist_numeric")
        
        # Preprocessing options
        st.markdown("---")
        st.subheader("⚙️ Preprocessing Options")
        
        handle_missing = st.selectbox(
            "Handle Missing Values",
            ["drop", "mean", "median", "mode"],
            help="Strategy for handling missing values"
        )
        
        scale_features = st.checkbox("Scale Numeric Features", value=True)
        encode_categorical = st.checkbox("Encode Categorical Features", value=True)
        
        if st.button("🔄 Apply Preprocessing"):
            with st.spinner("Preprocessing data..."):
                # Create a copy
                processed_df = df.copy()
                
                # Handle missing values
                if handle_missing == "drop":
                    processed_df = processed_df.dropna()
                elif handle_missing == "mean":
                    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].mean())
                elif handle_missing == "median":
                    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].median())
                elif handle_missing == "mode":
                    for col in processed_df.columns:
                        processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
                
                # Scale numeric features
                if scale_features:
                    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
                    if target_column in numeric_cols:
                        numeric_cols.remove(target_column)
                    
                    scaler = StandardScaler()
                    processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
                    st.session_state['dl_scaler'] = scaler
                
                # Encode categorical features
                if encode_categorical:
                    categorical_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
                    if target_column in categorical_cols:
                        categorical_cols.remove(target_column)
                    
                    label_encoders = {}
                    for col in categorical_cols:
                        le = LabelEncoder()
                        processed_df[col] = le.fit_transform(processed_df[col])
                        label_encoders[col] = le
                    
                    st.session_state['dl_encoders'] = label_encoders
                
                # Encode target if categorical
                if processed_df[target_column].dtype == 'object':
                    target_encoder = LabelEncoder()
                    processed_df[target_column] = target_encoder.fit_transform(processed_df[target_column])
                    st.session_state['dl_target_encoder'] = target_encoder
                
                st.session_state['dl_processed_data'] = processed_df
                st.success(f"✅ Data preprocessed! New shape: {processed_df.shape}")
                st.dataframe(processed_df.head(), use_container_width=True)
    
    @staticmethod
    def build_neural_network(df, target_column):
        """
        Build and configure neural network
        """
        st.subheader("🧠 Neural Network Architecture")
        
        # Check if data is preprocessed
        if 'dl_processed_data' not in st.session_state:
            st.warning("⚠️ Please preprocess data first in the Data Prep tab")
            return
        
        processed_df = st.session_state['dl_processed_data']
        
        # Determine task type
        is_classification = processed_df[target_column].nunique() < 10
        task_type = "Classification" if is_classification else "Regression"
        
        st.info(f"📝 Detected task type: **{task_type}**")
        
        # Network architecture
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏗️ Architecture")
            
            # Input layer
            input_features = [col for col in processed_df.columns if col != target_column]
            input_dim = len(input_features)
            
            st.write(f"Input Dimension: {input_dim}")
            
            # Hidden layers
            num_layers = st.slider("Number of Hidden Layers", 1, 5, 2)
            
            layers_config = []
            for i in range(num_layers):
                neurons = st.slider(f"Layer {i+1} - Neurons", 10, 500, 128 // (i+1))
                activation = st.selectbox(f"Layer {i+1} - Activation", 
                                       ["relu", "tanh", "sigmoid", "elu"], 
                                       key=f"activation_{i}")
                layers_config.append((neurons, activation))
            
            # Output layer
            if is_classification:
                output_neurons = processed_df[target_column].nunique()
                output_activation = "softmax"
                loss_function = "sparse_categorical_crossentropy"
                metrics = ["accuracy"]
            else:
                output_neurons = 1
                output_activation = "linear"
                loss_function = "mse"
                metrics = ["mae"]
        
        with col2:
            st.subheader("⚙️ Training Parameters")
            
            # Training parameters
            learning_rate = st.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=0)
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
            epochs = st.slider("Epochs", 10, 200, 50)
            validation_split = st.slider("Validation Split", 0.1, 0.4, 0.2)
            
            # Optimizer
            optimizer = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
            
            # Regularization
            use_dropout = st.checkbox("Use Dropout", value=True)
            dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 0.3) if use_dropout else 0.0
            
            use_early_stopping = st.checkbox("Use Early Stopping", value=True)
        
        # Build model button
        if st.button("🏗️ Build & Train Model", type="primary"):
            with st.spinner("Building and training model..."):
                try:
                    # Prepare data
                    X = processed_df[input_features].values
                    y = processed_df[target_column].values
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=validation_split, random_state=42
                    )
                    
                    # Build model
                    model = keras.Sequential()
                    
                    # Input layer
                    model.add(layers.Input(shape=(input_dim,)))
                    
                    # Hidden layers
                    for i, (neurons, activation) in enumerate(layers_config):
                        model.add(layers.Dense(neurons, activation=activation))
                        if use_dropout and i < len(layers_config) - 1:
                            model.add(layers.Dropout(dropout_rate))
                    
                    # Output layer
                    model.add(layers.Dense(output_neurons, activation=output_activation))
                    
                    # Compile model
                    model.compile(
                        optimizer=optimizer,
                        loss=loss_function,
                        metrics=metrics
                    )
                    
                    # Callbacks
                    callbacks = []
                    if use_early_stopping:
                        early_stopping = keras.callbacks.EarlyStopping(
                            monitor='val_loss' if not is_classification else 'val_accuracy',
                            patience=10,
                            restore_best_weights=True
                        )
                        callbacks.append(early_stopping)
                    
                    # Train model
                    history = model.fit(
                        X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split,
                        callbacks=callbacks,
                        verbose=0
                    )
                    
                    # Evaluate model
                    test_loss, test_metric = model.evaluate(X_test, y_test, verbose=0)
                    
                    # Store model and history
                    st.session_state['dl_model'] = model
                    st.session_state['dl_history'] = history
                    st.session_state['dl_test_results'] = {
                        'loss': test_loss,
                        'metric': test_metric,
                        'metric_name': metrics[0]
                    }
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Display model summary
                    st.subheader("📋 Model Summary")
                    model_summary = []
                    model.summary(print_fn=lambda x: model_summary.append(x))
                    st.text("\n".join(model_summary))
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Test Loss", f"{test_loss:.4f}")
                    with col2:
                        st.metric(f"Test {metrics[0].title()}", f"{test_metric:.4f}")
                    
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
    
    @staticmethod
    def training_history():
        """
        Display training history
        """
        st.subheader("📈 Training History")
        
        if 'dl_history' not in st.session_state:
            st.warning("⚠️ No training history available. Please train a model first.")
            return
        
        history = st.session_state['dl_history']
        
        # Create plots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Model Loss", "Model Metrics"),
            vertical_spacing=0.1
        )
        
        # Loss plot
        fig.add_trace(
            go.Scatter(y=history.history['loss'], name='Training Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=history.history['val_loss'], name='Validation Loss', line=dict(color='red')),
            row=1, col=1
        )
        
        # Metrics plot
        for metric_name in history.history:
            if 'loss' not in metric_name:
                fig.add_trace(
                    go.Scatter(y=history.history[metric_name], name=f'Training {metric_name}', line=dict(color='green')),
                    row=2, col=1
                )
                if f'val_{metric_name}' in history.history:
                    fig.add_trace(
                        go.Scatter(y=history.history[f'val_{metric_name}'], name=f'Validation {metric_name}', line=dict(color='orange')),
                        row=2, col=1
                    )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True, key="dl_training_history")
        
        # Training statistics
        st.subheader("📊 Training Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Epochs", len(history.history['loss']))
        
        with col2:
            best_loss = min(history.history['val_loss'])
            st.metric("Best Validation Loss", f"{best_loss:.4f}")
        
        with col3:
            if 'val_accuracy' in history.history:
                best_acc = max(history.history['val_accuracy'])
                st.metric("Best Validation Accuracy", f"{best_acc:.4f}")
    
    @staticmethod
    def model_evaluation(df, target_column):
        """
        Evaluate trained model
        """
        st.subheader("🎯 Model Evaluation")
        
        if 'dl_model' not in st.session_state:
            st.warning("⚠️ No trained model available. Please train a model first.")
            return
        
        model = st.session_state['dl_model']
        
        # Check if data is preprocessed
        if 'dl_processed_data' not in st.session_state:
            st.warning("⚠️ Please preprocess data first in the Data Prep tab")
            return
        
        processed_df = st.session_state['dl_processed_data']
        input_features = [col for col in processed_df.columns if col != target_column]
        
        # Prepare test data
        X = processed_df[input_features].values
        y = processed_df[target_column].values
        
        # Make predictions
        predictions = model.predict(X)
        
        # Determine task type
        is_classification = processed_df[target_column].nunique() < 10
        
        if is_classification:
            # Classification evaluation
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Metrics
            accuracy = accuracy_score(y, predicted_classes)
            
            st.subheader("📊 Classification Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
            
            with col2:
                st.metric("Error Rate", f"{1-accuracy:.4f}")
            
            # Classification report
            st.subheader("📋 Classification Report")
            
            # Get original labels if encoder exists
            if 'dl_target_encoder' in st.session_state:
                target_encoder = st.session_state['dl_target_encoder']
                original_labels = target_encoder.classes_
                
                report = classification_report(
                    y, predicted_classes, 
                    target_names=original_labels,
                    output_dict=True
                )
                
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(4), use_container_width=True)
            else:
                report = classification_report(y, predicted_classes, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(4), use_container_width=True)
            
            # Confusion matrix
            st.subheader("🔍 Confusion Matrix")
            cm = confusion_matrix(y, predicted_classes)
            
            fig = px.imshow(
                cm, 
                text_auto=True, 
                aspect="auto",
                title="Confusion Matrix"
            )
            st.plotly_chart(fig, use_container_width=True, key="dl_confusion_matrix")
            
        else:
            # Regression evaluation
            predictions = predictions.flatten()
            
            # Metrics
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, predictions)
            mae = np.mean(np.abs(y - predictions))
            
            st.subheader("📊 Regression Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MSE", f"{mse:.4f}")
            with col2:
                st.metric("RMSE", f"{rmse:.4f}")
            with col3:
                st.metric("R²", f"{r2:.4f}")
            with col4:
                st.metric("MAE", f"{mae:.4f}")
            
            # Prediction vs Actual plot
            st.subheader("📈 Predictions vs Actual")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y, y=predictions,
                mode='markers',
                name='Predictions',
                marker=dict(color='blue', opacity=0.6)
            ))
            
            # Perfect prediction line
            min_val = min(y.min(), predictions.min())
            max_val = max(y.max(), predictions.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title='Predictions vs Actual Values',
                xaxis_title='Actual Values',
                yaxis_title='Predicted Values'
            )
            
            st.plotly_chart(fig, use_container_width=True, key="dl_predictions_vs_actual")
            
            # Residuals plot
            st.subheader("📊 Residuals Analysis")
            
            residuals = y - predictions
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Residuals vs Predicted", "Residuals Distribution"),
                horizontal_spacing=0.1
            )
            
            # Residuals vs predicted
            fig.add_trace(
                go.Scatter(
                    x=predictions, y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(color='blue', opacity=0.6)
                ),
                row=1, col=1
            )
            
            # Residuals distribution
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    name='Residuals Distribution',
                    nbinsx=30
                ),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="dl_residuals_analysis")
        
        # Model saving option
        st.markdown("---")
        if st.button("💾 Save Model"):
            try:
                model.save("saved_models/deep_learning_model.h5")
                st.success("✅ Model saved successfully!")
            except Exception as e:
                st.error(f"❌ Error saving model: {str(e)}")

# Create instance for import compatibility
deep_learning = DeepLearningModels()