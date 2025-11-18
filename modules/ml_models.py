"""
Advanced ML Models Module for AI Data Analysis Platform
Comprehensive machine learning models with automated training, evaluation, and interpretation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, silhouette_score,
    classification_report, confusion_matrix
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE

# Advanced ML Libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
import shap

# Deep Learning (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from utils.helpers import DataValidator, PerformanceMonitor
except ImportError:
    class DataValidator:
        @staticmethod
        def validate_dataframe(df):
            return True
    
    class PerformanceMonitor:
        @staticmethod
        def measure_execution_time(func):
            return func(), 0

try:
    from utils.error_handler import handle_errors, MLModelError, ValidationError
except ImportError:
    def handle_errors(component_name):
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error in {component_name}: {str(e)}")
                    return None
            return wrapper
        return decorator
    
    class MLModelError(Exception):
        pass
    
    class ValidationError(Exception):
        pass

try:
    from utils.rate_limiter import rate_limit
except ImportError:
    def rate_limit():
        def decorator(func):
            return func
        return decorator


class MLModelManager:
    """Advanced machine learning model management system."""
    
    def __init__(self):
        """Initialize ML model manager."""
        self.data = None
        self.target_column = None
        self.feature_columns = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.model_results = {}
        self.best_model = None
        self.feature_importance = {}
        self.predictions = {}
        self.scalers = {}
        self.encoders = {}
        self.problem_type = None
        self.validator = DataValidator()
        self.performance_monitor = PerformanceMonitor()
    
    @handle_errors("ml")
    def setup_data(self, data: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """Setup data for machine learning."""
        if data is None or data.empty:
            raise ValidationError("No data provided")
        
        if target_column not in data.columns:
            raise ValidationError(f"Target column '{target_column}' not found in data")
        
        self.data = data.copy()
        self.target_column = target_column
        self.feature_columns = [col for col in data.columns if col != target_column]
        
        # Determine problem type
        self.problem_type = self._determine_problem_type(data[target_column])
        
        # Prepare features and target
        X = data[self.feature_columns]
        y = data[target_column]
        
        # Handle categorical variables
        X, y = self._prepare_data(X, y)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if self.problem_type == 'classification' else None
        )
        
        setup_info = {
            'problem_type': self.problem_type,
            'training_shape': self.X_train.shape,
            'testing_shape': self.X_test.shape,
            'feature_count': len(self.feature_columns),
            'target_distribution': dict(y.value_counts() if self.problem_type == 'classification' else y.describe())
        }
        
        return setup_info
    
    def _determine_problem_type(self, target_series: pd.Series) -> str:
        """Determine if it's classification or regression problem."""
        unique_values = target_series.nunique()
        data_type = target_series.dtype
        
        # Classification if:
        # 1. Less than 20 unique values OR
        # 2. Object/categorical data type OR
        # 3. Integer type with relatively few unique values
        if (unique_values < 20 or 
            data_type == 'object' or 
            (data_type in ['int64', 'int32'] and unique_values < len(target_series) * 0.05)):
            return 'classification'
        else:
            return 'regression'
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for ML (encoding, scaling, etc.)."""
        processed_X = X.copy()
        processed_y = y.copy()
        
        # Encode categorical features
        categorical_cols = processed_X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                processed_X[col] = self.encoders[col].fit_transform(processed_X[col].astype(str))
            else:
                processed_X[col] = self.encoders[col].transform(processed_X[col].astype(str))
        
        # Encode target if it's categorical
        if self.problem_type == 'classification' and processed_y.dtype == 'object':
            if 'target_encoder' not in self.encoders:
                self.encoders['target_encoder'] = LabelEncoder()
                processed_y = pd.Series(self.encoders['target_encoder'].fit_transform(processed_y))
            else:
                processed_y = pd.Series(self.encoders['target_encoder'].transform(processed_y))
        
        # Scale features
        if 'feature_scaler' not in self.scalers:
            self.scalers['feature_scaler'] = StandardScaler()
            processed_X = pd.DataFrame(
                self.scalers['feature_scaler'].fit_transform(processed_X),
                columns=processed_X.columns,
                index=processed_X.index
            )
        else:
            processed_X = pd.DataFrame(
                self.scalers['feature_scaler'].transform(processed_X),
                columns=processed_X.columns,
                index=processed_X.index
            )
        
        return processed_X, processed_y
    
    @handle_errors("ml")
    def train_models(self, model_types: List[str] = None, hyperparameter_tuning: bool = False) -> Dict[str, Any]:
        """Train multiple ML models and compare performance."""
        if self.X_train is None:
            raise ValidationError("Data not setup. Call setup_data() first.")
        
        if model_types is None:
            model_types = self._get_default_models()
        
        training_results = {}
        
        for model_type in model_types:
            try:
                model_result = self._train_single_model(model_type, hyperparameter_tuning)
                training_results[model_type] = model_result
            except Exception as e:
                training_results[model_type] = {'error': str(e)}
        
        # Find best model
        self.best_model = self._find_best_model(training_results)
        self.model_results = training_results
        
        return {
            'training_results': training_results,
            'best_model': self.best_model,
            'comparison_summary': self._create_model_comparison(training_results)
        }
    
    def _get_default_models(self) -> List[str]:
        """Get default models based on problem type."""
        if self.problem_type == 'classification':
            return [
                'logistic_regression', 'random_forest', 'gradient_boosting',
                'xgboost', 'lightgbm', 'catboost', 'svm', 'knn', 'naive_bayes'
            ]
        else:
            return [
                'linear_regression', 'random_forest', 'gradient_boosting',
                'xgboost', 'lightgbm', 'catboost', 'svm', 'ridge', 'lasso'
            ]
    
    def _train_single_model(self, model_type: str, hyperparameter_tuning: bool = False) -> Dict[str, Any]:
        """Train a single model."""
        model = self._get_model(model_type)
        
        if hyperparameter_tuning:
            model = self._tune_hyperparameters(model, model_type)
        
        # Train model
        start_time = datetime.now()
        model.fit(self.X_train, self.y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Make predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(self.y_train, y_pred_train)
        test_metrics = self._calculate_metrics(self.y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = self._cross_validate_model(model)
        
        # Store model and results
        self.models[model_type] = model
        self.predictions[model_type] = {
            'train': y_pred_train,
            'test': y_pred_test
        }
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model_type] = dict(zip(self.feature_columns, model.feature_importances_))
        
        return {
            'model': model,
            'training_time': training_time,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'feature_importance': self.feature_importance.get(model_type, {})
        }
    
    def _get_model(self, model_type: str):
        """Get model instance based on type."""
        if self.problem_type == 'classification':
            models = {
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
                'gradient_boosting': GradientBoostingClassifier(random_state=42),
                'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'catboost': CatBoostClassifier(random_state=42, verbose=False),
                'svm': SVC(random_state=42, probability=True),
                'knn': KNeighborsClassifier(),
                'naive_bayes': GaussianNB(),
                'decision_tree': DecisionTreeClassifier(random_state=42)
            }
        else:
            models = {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(random_state=42, n_estimators=100),
                'gradient_boosting': GradientBoostingRegressor(random_state=42),
                'xgboost': xgb.XGBRegressor(random_state=42, eval_metric='rmse'),
                'lightgbm': lgb.LGBMRegressor(random_state=42, verbose=-1),
                'catboost': CatBoostRegressor(random_state=42, verbose=False),
                'svm': SVR(),
                'knn': KNeighborsRegressor(),
                'ridge': Ridge(random_state=42),
                'lasso': Lasso(random_state=42),
                'elastic_net': ElasticNet(random_state=42),
                'decision_tree': DecisionTreeRegressor(random_state=42)
            }
        
        if model_type not in models:
            raise MLModelError(f"Unknown model type: {model_type}")
        
        return models[model_type]
    
    def _tune_hyperparameters(self, model, model_type: str):
        """Tune hyperparameters using GridSearchCV."""
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear']
            }
        }
        
        if model_type in param_grids:
            grid_search = GridSearchCV(
                model, param_grids[model_type], 
                cv=3, scoring='accuracy' if self.problem_type == 'classification' else 'r2',
                n_jobs=-1
            )
            grid_search.fit(self.X_train, self.y_train)
            return grid_search.best_estimator_
        
        return model
    
    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate performance metrics."""
        if self.problem_type == 'classification':
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
        else:
            return {
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
            }
    
    def _cross_validate_model(self, model) -> Dict[str, float]:
        """Perform cross-validation."""
        scoring = 'accuracy' if self.problem_type == 'classification' else 'r2'
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring=scoring)
        
        return {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
    
    def _find_best_model(self, training_results: Dict[str, Any]) -> str:
        """Find the best model based on performance."""
        best_score = -np.inf if self.problem_type == 'classification' else -np.inf
        best_model = None
        
        for model_type, result in training_results.items():
            if 'error' in result:
                continue
            
            if self.problem_type == 'classification':
                score = result['test_metrics']['accuracy']
            else:
                score = result['test_metrics']['r2']
            
            if score > best_score:
                best_score = score
                best_model = model_type
        
        return best_model
    
    def _create_model_comparison(self, training_results: Dict[str, Any]) -> pd.DataFrame:
        """Create comparison table of all models."""
        comparison_data = []
        
        for model_type, result in training_results.items():
            if 'error' in result:
                continue
            
            row = {'Model': model_type}
            
            if self.problem_type == 'classification':
                row.update({
                    'Accuracy': result['test_metrics']['accuracy'],
                    'Precision': result['test_metrics']['precision'],
                    'Recall': result['test_metrics']['recall'],
                    'F1-Score': result['test_metrics']['f1'],
                    'CV_Mean': result['cv_scores']['mean'],
                    'Training_Time': result['training_time']
                })
            else:
                row.update({
                    'R²': result['test_metrics']['r2'],
                    'RMSE': result['test_metrics']['rmse'],
                    'MAE': result['test_metrics']['mae'],
                    'CV_Mean': result['cv_scores']['mean'],
                    'Training_Time': result['training_time']
                })
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data).sort_values(
            by='Accuracy' if self.problem_type == 'classification' else 'R²',
            ascending=False
        )
    
    @handle_errors("ml")
    def perform_feature_selection(self, method: str = 'auto', k_best: int = 10) -> Dict[str, Any]:
        """Perform feature selection."""
        if self.X_train is None:
            raise ValidationError("Data not setup. Call setup_data() first.")
        
        feature_selection_results = {}
        
        if method == 'auto':
            methods = ['univariate', 'rfe', 'importance']
        else:
            methods = [method]
        
        for sel_method in methods:
            try:
                if sel_method == 'univariate':
                    selected_features = self._univariate_feature_selection(k_best)
                elif sel_method == 'rfe':
                    selected_features = self._rfe_feature_selection(k_best)
                elif sel_method == 'importance':
                    selected_features = self._importance_feature_selection(k_best)
                else:
                    continue
                
                feature_selection_results[sel_method] = selected_features
            except Exception as e:
                feature_selection_results[sel_method] = {'error': str(e)}
        
        return feature_selection_results
    
    def _univariate_feature_selection(self, k_best: int) -> Dict[str, Any]:
        """Univariate feature selection."""
        if self.problem_type == 'classification':
            selector = SelectKBest(score_func=f_classif, k=k_best)
        else:
            selector = SelectKBest(score_func=f_regression, k=k_best)
        
        selector.fit(self.X_train, self.y_train)
        
        selected_features = [self.feature_columns[i] for i in selector.get_support(indices=True)]
        feature_scores = dict(zip(self.feature_columns, selector.scores_))
        
        return {
            'selected_features': selected_features,
            'feature_scores': feature_scores,
            'selector': selector
        }
    
    def _rfe_feature_selection(self, k_best: int) -> Dict[str, Any]:
        """Recursive Feature Elimination."""
        estimator = RandomForestClassifier(n_estimators=50, random_state=42) if self.problem_type == 'classification' else RandomForestRegressor(n_estimators=50, random_state=42)
        
        selector = RFE(estimator=estimator, n_features_to_select=k_best)
        selector.fit(self.X_train, self.y_train)
        
        selected_features = [self.feature_columns[i] for i in selector.get_support(indices=True)]
        feature_ranking = dict(zip(self.feature_columns, selector.ranking_))
        
        return {
            'selected_features': selected_features,
            'feature_ranking': feature_ranking,
            'selector': selector
        }
    
    def _importance_feature_selection(self, k_best: int) -> Dict[str, Any]:
        """Feature selection based on model importance."""
        if self.best_model and self.best_model in self.feature_importance:
            importance_scores = self.feature_importance[self.best_model]
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feat for feat, _ in sorted_features[:k_best]]
            
            return {
                'selected_features': selected_features,
                'importance_scores': importance_scores,
                'model_used': self.best_model
            }
        else:
            # Train a quick random forest for importance
            rf = RandomForestClassifier(n_estimators=50, random_state=42) if self.problem_type == 'classification' else RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(self.X_train, self.y_train)
            
            importance_scores = dict(zip(self.feature_columns, rf.feature_importances_))
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feat for feat, _ in sorted_features[:k_best]]
            
            return {
                'selected_features': selected_features,
                'importance_scores': importance_scores,
                'model_used': 'random_forest'
            }
    
    @handle_errors("ml")
    def perform_clustering(self, algorithm: str = 'kmeans', n_clusters: int = 3) -> Dict[str, Any]:
        """Perform clustering analysis."""
        if self.X_train is None:
            raise ValidationError("Data not setup. Call setup_data() first.")
        
        clustering_results = {}
        
        if algorithm == 'kmeans':
            clustering_results = self._kmeans_clustering(n_clusters)
        elif algorithm == 'dbscan':
            clustering_results = self._dbscan_clustering()
        elif algorithm == 'hierarchical':
            clustering_results = self._hierarchical_clustering(n_clusters)
        
        return clustering_results
    
    def _kmeans_clustering(self, n_clusters: int) -> Dict[str, Any]:
        """K-Means clustering."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.X_train)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(self.X_train, cluster_labels)
        
        # Get cluster centers
        cluster_centers = kmeans.cluster_centers_
        
        return {
            'algorithm': 'kmeans',
            'cluster_labels': cluster_labels,
            'silhouette_score': silhouette_avg,
            'cluster_centers': cluster_centers,
            'inertia': kmeans.inertia_,
            'model': kmeans
        }
    
    def _dbscan_clustering(self) -> Dict[str, Any]:
        """DBSCAN clustering."""
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(self.X_train)
        
        # Calculate silhouette score (only if more than 1 cluster)
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(self.X_train, cluster_labels)
        else:
            silhouette_avg = -1
        
        return {
            'algorithm': 'dbscan',
            'cluster_labels': cluster_labels,
            'silhouette_score': silhouette_avg,
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'n_noise': list(cluster_labels).count(-1),
            'model': dbscan
        }
    
    def _hierarchical_clustering(self, n_clusters: int) -> Dict[str, Any]:
        """Hierarchical clustering."""
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = hierarchical.fit_predict(self.X_train)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(self.X_train, cluster_labels)
        
        return {
            'algorithm': 'hierarchical',
            'cluster_labels': cluster_labels,
            'silhouette_score': silhouette_avg,
            'n_clusters': n_clusters,
            'model': hierarchical
        }
    
    @handle_errors("ml")
    def generate_model_explanations(self, model_type: str = None, sample_size: int = 100) -> Dict[str, Any]:
        """Generate model explanations using SHAP."""
        if model_type is None:
            model_type = self.best_model
        
        if model_type is None or model_type not in self.models:
            raise ValidationError("No trained model available for explanation")
        
        model = self.models[model_type]
        
        try:
            # Create SHAP explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model, self.X_train)
            else:
                explainer = shap.Explainer(model, self.X_train)
            
            # Calculate SHAP values
            shap_values = explainer(self.X_test[:sample_size])
            
            # Feature importance based on SHAP
            shap_importance = np.abs(shap_values.values).mean(axis=0)
            feature_names = self.feature_columns
            
            shap_summary = dict(zip(feature_names, shap_importance))
            
            return {
                'model_type': model_type,
                'shap_values': shap_values,
                'feature_importance': shap_summary,
                'explainer': explainer,
                'sample_data': self.X_test[:sample_size]
            }
        
        except Exception as e:
            return {'error': f"SHAP explanation failed: {str(e)}"}
    
    @handle_errors("ml")
    def create_ensemble_model(self, voting_method: str = 'soft') -> Dict[str, Any]:
        """Create ensemble model from best performing models."""
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        
        if self.problem_type == 'classification':
            # Get top 3 models
            top_models = self._get_top_models(3)
            
            if len(top_models) < 2:
                raise ValidationError("Need at least 2 models for ensemble")
            
            estimators = [(name, self.models[name]) for name in top_models]
            ensemble = VotingClassifier(estimators=estimators, voting=voting_method)
        else:
            top_models = self._get_top_models(3)
            
            if len(top_models) < 2:
                raise ValidationError("Need at least 2 models for ensemble")
            
            estimators = [(name, self.models[name]) for name in top_models]
            ensemble = VotingRegressor(estimators=estimators)
        
        # Train ensemble
        ensemble.fit(self.X_train, self.y_train)
        
        # Evaluate ensemble
        y_pred = ensemble.predict(self.X_test)
        metrics = self._calculate_metrics(self.y_test, y_pred)
        
        # Store ensemble model
        self.models['ensemble'] = ensemble
        
        return {
            'ensemble_model': ensemble,
            'base_models': top_models,
            'voting_method': voting_method,
            'metrics': metrics,
            'predictions': y_pred
        }
    
    def _get_top_models(self, n: int) -> List[str]:
        """Get top N performing models."""
        model_scores = {}
        
        for model_type, result in self.model_results.items():
            if 'error' in result:
                continue
            
            if self.problem_type == 'classification':
                score = result['test_metrics']['accuracy']
            else:
                score = result['test_metrics']['r2']
            
            model_scores[model_type] = score
        
        # Sort by score and get top N
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        return [model for model, _ in sorted_models[:n]]
    
    @rate_limit()
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        return {
            'problem_type': self.problem_type,
            'data_info': {
                'features': self.feature_columns,
                'target': self.target_column,
                'training_samples': len(self.X_train) if self.X_train is not None else 0,
                'testing_samples': len(self.X_test) if self.X_test is not None else 0
            },
            'trained_models': list(self.models.keys()),
            'best_model': self.best_model,
            'model_results': self.model_results,
            'feature_importance': self.feature_importance,
            'performance_comparison': self._create_model_comparison(self.model_results).to_dict() if self.model_results else {}
        }
    
    def save_model(self, model_type: str, filepath: str) -> str:
        """Save trained model to file."""
        if model_type not in self.models:
            raise ValidationError(f"Model '{model_type}' not found")
        
        import joblib
        
        model_data = {
            'model': self.models[model_type],
            'scaler': self.scalers.get('feature_scaler'),
            'encoders': self.encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'problem_type': self.problem_type
        }
        
        joblib.dump(model_data, filepath)
        return filepath
    
    def load_model(self, filepath: str) -> Dict[str, Any]:
        """Load trained model from file."""
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.models['loaded_model'] = model_data['model']
        self.scalers = model_data.get('scaler', {})
        self.encoders = model_data.get('encoders', {})
        self.feature_columns = model_data.get('feature_columns', [])
        self.target_column = model_data.get('target_column', '')
        self.problem_type = model_data.get('problem_type', '')
        
        return model_data
    
    def reset_models(self):
        """Reset all trained models."""
        self.models = {}
        self.model_results = {}
        self.best_model = None
        self.feature_importance = {}
        self.predictions = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


class AutoML:
    """Automated Machine Learning for quick model building."""
    
    def __init__(self):
        """Initialize AutoML."""
        self.ml_manager = MLModelManager()
        self.automl_results = {}
    
    @handle_errors("ml")
    def auto_train(self, data: pd.DataFrame, target_column: str, time_limit: int = 300) -> Dict[str, Any]:
        """Automatically train and select best models."""
        import time
        
        start_time = time.time()
        
        # Setup data
        setup_info = self.ml_manager.setup_data(data, target_column)
        
        # Quick model training
        if self.ml_manager.problem_type == 'classification':
            quick_models = ['logistic_regression', 'random_forest', 'xgboost']
        else:
            quick_models = ['linear_regression', 'random_forest', 'xgboost']
        
        # Train models with time limit
        training_results = self.ml_manager.train_models(quick_models)
        
        # Feature selection
        feature_selection = self.ml_manager.perform_feature_selection(k_best=min(20, len(self.ml_manager.feature_columns)))
        
        # Create ensemble
        try:
            ensemble_results = self.ml_manager.create_ensemble_model()
        except:
            ensemble_results = {'error': 'Ensemble creation failed'}
        
        # Generate explanations for best model
        explanations = {}
        if self.ml_manager.best_model:
            try:
                explanations = self.ml_manager.generate_model_explanations()
            except:
                explanations = {'error': 'Explanation generation failed'}
        
        total_time = time.time() - start_time
        
        self.automl_results = {
            'setup_info': setup_info,
            'training_results': training_results,
            'feature_selection': feature_selection,
            'ensemble_results': ensemble_results,
            'explanations': explanations,
            'total_time': total_time,
            'best_model': self.ml_manager.best_model
        }
        
        return self.automl_results
    
    def get_automl_summary(self) -> Dict[str, Any]:
        """Get AutoML summary."""
        return self.automl_results


# Standalone functions for backward compatibility
def train_classification_models(df, target_column, model_types=None):
    """
    Train classification models (standalone function)
    """
    ml_manager = MLModelManager()
    setup_info = ml_manager.setup_data(df, target_column)
    
    if ml_manager.problem_type != 'classification':
        return {'error': 'This is not a classification problem'}
    
    if model_types is None:
        model_types = ['random_forest', 'xgboost', 'logistic', 'lightgbm']
    
    results = ml_manager.train_models(model_types)
    
    # Convert to expected format
    formatted_results = {}
    for model_type, result in results['training_results'].items():
        if 'error' not in result:
            formatted_results[model_type] = {
                'model': result['model'],
                'metrics': result['test_metrics'],
                'predictions': result['predictions']['test']
            }
    
    return formatted_results


def train_regression_models(df, target_column, model_types=None):
    """
    Train regression models (standalone function)
    """
    ml_manager = MLModelManager()
    setup_info = ml_manager.setup_data(df, target_column)
    
    if ml_manager.problem_type != 'regression':
        return {'error': 'This is not a regression problem'}
    
    if model_types is None:
        model_types = ['random_forest', 'xgboost', 'ridge', 'lasso']
    
    results = ml_manager.train_models(model_types)
    
    # Convert to expected format
    formatted_results = {}
    for model_type, result in results['training_results'].items():
        if 'error' not in result:
            formatted_results[model_type] = {
                'model': result['model'],
                'metrics': result['test_metrics'],
                'predictions': result['predictions']['test']
            }
    
    return formatted_results


def perform_clustering(df, n_clusters=3, method='kmeans'):
    """
    Perform clustering analysis (standalone function)
    """
    ml_manager = MLModelManager()
    
    # Use all columns as features for clustering
    temp_target = 'temp_target_for_clustering'
    df_temp = df.copy()
    df_temp[temp_target] = 0  # Dummy target
    
    setup_info = ml_manager.setup_data(df_temp, temp_target)
    
    results = ml_manager.perform_clustering(method, n_clusters)
    
    if 'error' not in results:
        return {
            'n_clusters': results.get('n_clusters', n_clusters),
            'labels': results['cluster_labels'],
            'silhouette_score': results.get('silhouette_score', 0)
        }
    else:
        return results


def perform_pca(df, n_components=2):
    """
    Perform PCA analysis (standalone function)
    """
    try:
        # Prepare data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols].dropna()
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create results
        results = {
            'transformed_data': X_pca,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_,
            'feature_names': numeric_cols.tolist()
        }
        
        return results
    except Exception as e:
        return {'error': str(e)}


def get_feature_importance(model, feature_names):
    """
    Get feature importance from a trained model (standalone function)
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
            if len(importance.shape) > 1:
                importance = importance.mean(axis=0)
        else:
            return None
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    except Exception as e:
        return None