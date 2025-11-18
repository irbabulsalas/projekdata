import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import pickle
import os

def train_neural_network(X_train, y_train, X_test, y_test, task_type="classification", 
                        hidden_layers=(100, 50), activation="relu", max_iter=500):
    """
    Train a neural network model
    """
    try:
        if task_type == "classification":
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                max_iter=max_iter,
                random_state=42
            )
        else:
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                max_iter=max_iter,
                random_state=42
            )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if task_type == "classification":
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred)
            }
        else:
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred
        }
    except Exception as e:
        st.error(f"Error training neural network: {str(e)}")
        return None

def train_xgboost(X_train, y_train, X_test, y_test, task_type="classification", params=None):
    """
    Train an XGBoost model
    """
    try:
        if params is None:
            params = {
                'max_depth': 6,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'random_state': 42
            }
        
        if task_type == "classification":
            model = xgb.XGBClassifier(**params)
        else:
            model = xgb.XGBRegressor(**params)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if task_type == "classification":
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred)
            }
        else:
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'feature_importance': feature_importance
        }
    except Exception as e:
        st.error(f"Error training XGBoost: {str(e)}")
        return None

def train_lightgbm(X_train, y_train, X_test, y_test, task_type="classification", params=None):
    """
    Train a LightGBM model
    """
    try:
        if params is None:
            params = {
                'max_depth': 6,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbosity': -1
            }
        
        if task_type == "classification":
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMRegressor(**params)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if task_type == "classification":
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred)
            }
        else:
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'feature_importance': feature_importance
        }
    except Exception as e:
        st.error(f"Error training LightGBM: {str(e)}")
        return None

def ensemble_models(models_dict, X_test, y_test, task_type="classification"):
    """
    Create an ensemble of multiple models
    """
    try:
        if not models_dict:
            st.error("No models provided for ensembling")
            return None
        
        # Get predictions from all models
        all_predictions = []
        for name, model_data in models_dict.items():
            if 'model' in model_data:
                model = model_data['model']
                pred = model.predict(X_test)
                all_predictions.append(pred)
        
        if not all_predictions:
            st.error("No valid models for ensembling")
            return None
        
        # Average predictions for regression, majority vote for classification
        if task_type == "classification":
            # Majority vote
            from scipy.stats import mode
            ensemble_pred = mode(all_predictions, axis=0)[0].flatten()
            metrics = {
                'accuracy': accuracy_score(y_test, ensemble_pred)
            }
        else:
            # Average predictions
            ensemble_pred = np.mean(all_predictions, axis=0)
            metrics = {
                'mse': mean_squared_error(y_test, ensemble_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
                'r2': r2_score(y_test, ensemble_pred)
            }
        
        return {
            'predictions': ensemble_pred,
            'metrics': metrics,
            'models_used': list(models_dict.keys())
        }
    except Exception as e:
        st.error(f"Error creating ensemble: {str(e)}")
        return None

def save_model(model, model_name, model_type="classification"):
    """
    Save a trained model to disk
    """
    try:
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
        
        filename = f"saved_models/{model_name}_{model_type}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        
        return filename
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        return None

def load_model(filename):
    """
    Load a saved model from disk
    """
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None