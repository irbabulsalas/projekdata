"""
Data Processing Module for AI Data Analysis Platform
Comprehensive data processing, cleaning, and transformation utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

from utils.helpers import DataTypeDetector, DataValidator, TextProcessor
from utils.error_handler import handle_errors, DataProcessingError, ValidationError
from utils.rate_limiter import rate_limit


class DataProcessor:
    """Advanced data processing and transformation utilities."""
    
    def __init__(self):
        """Initialize data processor."""
        self.data = None
        self.original_data = None
        self.processing_log = []
        self.type_detector = DataTypeDetector()
        self.validator = DataValidator()
        self.text_processor = TextProcessor()
    
    @handle_errors("data")
    def load_data(self, data_source: Union[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """Load data from various sources."""
        if isinstance(data_source, str):
            if data_source.endswith('.csv'):
                self.data = pd.read_csv(data_source, **kwargs)
            elif data_source.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(data_source, **kwargs)
            elif data_source.endswith('.json'):
                self.data = pd.read_json(data_source, **kwargs)
            elif data_source.endswith('.parquet'):
                self.data = pd.read_parquet(data_source, **kwargs)
            elif data_source.endswith('.tsv'):
                self.data = pd.read_csv(data_source, sep='\t', **kwargs)
            else:
                raise ValidationError(f"Unsupported file format: {data_source}")
        elif isinstance(data_source, pd.DataFrame):
            self.data = data_source.copy()
        else:
            raise ValidationError("Data source must be file path or DataFrame")
        
        self.original_data = self.data.copy()
        self._log_processing("Data loaded", {'shape': self.data.shape})
        
        return self.data
    
    def _log_processing(self, operation: str, details: Dict[str, Any] = None):
        """Log processing operations."""
        log_entry = {
            'timestamp': datetime.now(),
            'operation': operation,
            'details': details or {}
        }
        self.processing_log.append(log_entry)
    
    @handle_errors("data")
    def clean_data(self, cleaning_config: Dict[str, Any] = None) -> pd.DataFrame:
        """Comprehensive data cleaning with configurable options."""
        if self.data is None:
            raise ValidationError("No data loaded")
        
        config = cleaning_config or self._get_default_cleaning_config()
        
        # Handle missing values
        if config.get('handle_missing', True):
            self.data = self._handle_missing_values(config.get('missing_strategy', 'auto'))
        
        # Handle duplicates
        if config.get('handle_duplicates', True):
            self.data = self._handle_duplicates(config.get('duplicate_strategy', 'drop'))
        
        # Handle outliers
        if config.get('handle_outliers', True):
            self.data = self._handle_outliers(config.get('outlier_strategy', 'iqr'))
        
        # Standardize data types
        if config.get('standardize_types', True):
            self.data = self._standardize_data_types()
        
        # Clean text data
        if config.get('clean_text', True):
            self.data = self._clean_text_columns(config.get('text_config', {}))
        
        self._log_processing("Data cleaning completed", config)
        return self.data
    
    def _get_default_cleaning_config(self) -> Dict[str, Any]:
        """Get default cleaning configuration."""
        return {
            'handle_missing': True,
            'missing_strategy': 'auto',
            'handle_duplicates': True,
            'duplicate_strategy': 'drop',
            'handle_outliers': True,
            'outlier_strategy': 'iqr',
            'standardize_types': True,
            'clean_text': True,
            'text_config': {
                'remove_special_chars': False,
                'lowercase': True,
                'remove_numbers': False
            }
        }
    
    def _handle_missing_values(self, strategy: str = 'auto') -> pd.DataFrame:
        """Handle missing values with various strategies."""
        processed_data = self.data.copy()
        
        for col in processed_data.columns:
            missing_count = processed_data[col].isnull().sum()
            if missing_count == 0:
                continue
            
            missing_pct = (missing_count / len(processed_data)) * 100
            
            if strategy == 'auto':
                if missing_pct > 50:
                    processed_data = processed_data.drop(columns=[col])
                    self._log_processing(f"Dropped column {col} (>{missing_pct:.1f}% missing)")
                elif missing_pct > 20:
                    processed_data[col] = self._impute_column(processed_data[col], 'advanced')
                else:
                    processed_data[col] = self._impute_column(processed_data[col], 'simple')
            elif strategy == 'drop':
                processed_data = processed_data.dropna(subset=[col])
            elif strategy == 'impute':
                processed_data[col] = self._impute_column(processed_data[col], 'simple')
        
        return processed_data
    
    def _impute_column(self, series: pd.Series, method: str) -> pd.Series:
        """Impute missing values in a column."""
        if pd.api.types.is_numeric_dtype(series):
            if method == 'advanced':
                # Use median for skewed data, mean for normal data
                skewness = series.skew()
                if abs(skewness) > 1:
                    return series.fillna(series.median())
                else:
                    return series.fillna(series.mean())
            else:
                return series.fillna(series.mean())
        else:
            # For categorical data
            if series.mode().empty:
                return series.fillna('Unknown')
            return series.fillna(series.mode().iloc[0])
    
    def _handle_duplicates(self, strategy: str = 'drop') -> pd.DataFrame:
        """Handle duplicate records."""
        if strategy == 'drop':
            duplicate_count = self.data.duplicated().sum()
            if duplicate_count > 0:
                processed_data = self.data.drop_duplicates()
                self._log_processing(f"Removed {duplicate_count} duplicate records")
                return processed_data
        
        return self.data
    
    def _handle_outliers(self, strategy: str = 'iqr') -> pd.DataFrame:
        """Handle outliers in numeric columns."""
        processed_data = self.data.copy()
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if strategy == 'iqr':
                processed_data[col] = self._cap_outliers_iqr(processed_data[col])
            elif strategy == 'zscore':
                processed_data[col] = self._cap_outliers_zscore(processed_data[col])
            elif strategy == 'remove':
                processed_data = self._remove_outliers(processed_data, col)
        
        return processed_data
    
    def _cap_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Cap outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return series.clip(lower=lower_bound, upper=upper_bound)
    
    def _cap_outliers_zscore(self, series: pd.Series) -> pd.Series:
        """Cap outliers using Z-score method."""
        mean = series.mean()
        std = series.std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        
        return series.clip(lower=lower_bound, upper=upper_bound)
    
    def _remove_outliers(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Remove outliers from a specific column."""
        series = data[column].dropna()
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            self._log_processing(f"Removed {outlier_count} outliers from {column}")
            return data[~outlier_mask]
        
        return data
    
    def _standardize_data_types(self) -> pd.DataFrame:
        """Standardize and optimize data types."""
        processed_data = self.data.copy()
        
        for col in processed_data.columns:
            # Try to convert to datetime
            if processed_data[col].dtype == 'object':
                try:
                    processed_data[col] = pd.to_datetime(processed_data[col], errors='raise')
                    self._log_processing(f"Converted {col} to datetime")
                    continue
                except:
                    pass
                
                # Try to convert to numeric
                try:
                    processed_data[col] = pd.to_numeric(processed_data[col], errors='raise')
                    self._log_processing(f"Converted {col} to numeric")
                    continue
                except:
                    pass
            
            # Optimize numeric types
            if pd.api.types.is_numeric_dtype(processed_data[col]):
                processed_data[col] = self._optimize_numeric_type(processed_data[col])
            
            # Convert to category if appropriate
            elif processed_data[col].dtype == 'object':
                unique_ratio = processed_data[col].nunique() / len(processed_data)
                if unique_ratio < 0.1 and processed_data[col].nunique() <= 50:
                    processed_data[col] = processed_data[col].astype('category')
                    self._log_processing(f"Converted {col} to category")
        
        return processed_data
    
    def _optimize_numeric_type(self, series: pd.Series) -> pd.Series:
        """Optimize numeric data types for memory efficiency."""
        if pd.api.types.is_integer_dtype(series):
            if series.min() >= 0:
                if series.max() < 255:
                    return series.astype('uint8')
                elif series.max() < 65535:
                    return series.astype('uint16')
                elif series.max() < 4294967295:
                    return series.astype('uint32')
            else:
                if series.min() >= -128 and series.max() <= 127:
                    return series.astype('int8')
                elif series.min() >= -32768 and series.max() <= 32767:
                    return series.astype('int16')
                elif series.min() >= -2147483648 and series.max() <= 2147483647:
                    return series.astype('int32')
        elif pd.api.types.is_float_dtype(series):
            return series.astype('float32')
        
        return series
    
    def _clean_text_columns(self, text_config: Dict[str, Any]) -> pd.DataFrame:
        """Clean text columns."""
        processed_data = self.data.copy()
        text_cols = processed_data.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            # Skip if column looks like it contains structured data
            if self._is_structured_column(processed_data[col]):
                continue
            
            # Apply text cleaning
            processed_data[col] = processed_data[col].apply(
                lambda x: self.text_processor.clean_text(
                    x, 
                    remove_special_chars=text_config.get('remove_special_chars', False),
                    remove_numbers=text_config.get('remove_numbers', False),
                    lowercase=text_config.get('lowercase', True)
                ) if pd.notna(x) else x
            )
        
        return processed_data
    
    def _is_structured_column(self, series: pd.Series) -> bool:
        """Check if column contains structured data (not free text)."""
        sample = series.dropna().head(100)
        
        # Check for patterns that suggest structured data
        for value in sample:
            str_value = str(value)
            
            # Email pattern
            if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', str_value):
                return True
            
            # Phone pattern
            if re.match(r'^[\d\s\-\+\(\)]+$', str_value) and len(str_value) > 7:
                return True
            
            # ID pattern (alphanumeric with specific format)
            if re.match(r'^[A-Z]{2,4}\d{4,8}$', str_value):
                return True
        
        return False
    
    @handle_errors("data")
    def transform_data(self, transformation_config: Dict[str, Any] = None) -> pd.DataFrame:
        """Apply data transformations."""
        if self.data is None:
            raise ValidationError("No data loaded")
        
        config = transformation_config or {}
        
        # Feature engineering
        if config.get('feature_engineering', True):
            self.data = self._apply_feature_engineering(config.get('feature_config', {}))
        
        # Encoding categorical variables
        if config.get('encode_categorical', True):
            self.data = self._encode_categorical_variables(config.get('encoding_strategy', 'auto'))
        
        # Scaling numeric variables
        if config.get('scale_numeric', False):
            self.data = self._scale_numeric_variables(config.get('scaling_method', 'standard'))
        
        # Date/time features
        if config.get('extract_datetime_features', True):
            self.data = self._extract_datetime_features()
        
        self._log_processing("Data transformation completed", config)
        return self.data
    
    def _apply_feature_engineering(self, feature_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply feature engineering techniques."""
        processed_data = self.data.copy()
        
        # Polynomial features for numeric columns
        if feature_config.get('polynomial_features', False):
            processed_data = self._create_polynomial_features(processed_data)
        
        # Interaction features
        if feature_config.get('interaction_features', False):
            processed_data = self._create_interaction_features(processed_data)
        
        # Binning features
        if feature_config.get('binning_features', False):
            processed_data = self._create_binned_features(processed_data)
        
        # Text features
        if feature_config.get('text_features', False):
            processed_data = self._create_text_features(processed_data)
        
        return processed_data
    
    def _create_polynomial_features(self, data: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for numeric columns."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return data
        
        processed_data = data.copy()
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # Multiplication feature
                feature_name = f"{col1}_x_{col2}"
                processed_data[feature_name] = data[col1] * data[col2]
        
        return processed_data
    
    def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between categorical and numeric variables."""
        processed_data = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        for cat_col in categorical_cols[:3]:  # Limit to avoid too many features
            for num_col in numeric_cols[:3]:  # Limit to avoid too many features
                if data[cat_col].nunique() <= 10:  # Only for low cardinality
                    feature_name = f"{cat_col}_{num_col}_mean"
                    processed_data[feature_name] = data.groupby(cat_col)[num_col].transform('mean')
        
        return processed_data
    
    def _create_binned_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create binned features for numeric variables."""
        processed_data = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if data[col].nunique() > 10:  # Only bin continuous variables
                feature_name = f"{col}_binned"
                processed_data[feature_name] = pd.cut(data[col], bins=5, labels=False)
        
        return processed_data
    
    def _create_text_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features from text columns."""
        processed_data = data.copy()
        text_cols = data.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            if self._is_text_column(data[col]):
                # Text length
                processed_data[f"{col}_length"] = data[col].astype(str).str.len()
                
                # Word count
                processed_data[f"{col}_word_count"] = data[col].astype(str).str.split().str.len()
                
                # Character count (excluding spaces)
                processed_data[f"{col}_char_count"] = data[col].astype(str).str.replace(' ', '').str.len()
        
        return processed_data
    
    def _is_text_column(self, series: pd.Series) -> bool:
        """Check if column contains free text."""
        sample = series.dropna().head(50)
        if sample.empty:
            return False
        
        avg_length = sample.astype(str).str.len().mean()
        unique_ratio = series.nunique() / len(series)
        
        return avg_length > 20 and unique_ratio > 0.5
    
    def _encode_categorical_variables(self, strategy: str = 'auto') -> pd.DataFrame:
        """Encode categorical variables."""
        processed_data = self.data.copy()
        categorical_cols = processed_data.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            unique_count = processed_data[col].nunique()
            
            if strategy == 'auto':
                if unique_count == 2:
                    # Binary encoding
                    processed_data[col] = pd.factorize(processed_data[col])[0]
                elif unique_count <= 10:
                    # One-hot encoding
                    dummies = pd.get_dummies(processed_data[col], prefix=col)
                    processed_data = pd.concat([processed_data, dummies], axis=1)
                    processed_data = processed_data.drop(columns=[col])
                else:
                    # Label encoding
                    processed_data[col] = pd.factorize(processed_data[col])[0]
            elif strategy == 'onehot':
                dummies = pd.get_dummies(processed_data[col], prefix=col)
                processed_data = pd.concat([processed_data, dummies], axis=1)
                processed_data = processed_data.drop(columns=[col])
            elif strategy == 'label':
                processed_data[col] = pd.factorize(processed_data[col])[0]
        
        return processed_data
    
    def _scale_numeric_variables(self, method: str = 'standard') -> pd.DataFrame:
        """Scale numeric variables."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        processed_data = self.data.copy()
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            return processed_data
        
        processed_data[numeric_cols] = scaler.fit_transform(processed_data[numeric_cols])
        
        return processed_data
    
    def _extract_datetime_features(self) -> pd.DataFrame:
        """Extract features from datetime columns."""
        processed_data = self.data.copy()
        datetime_cols = processed_data.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_cols:
            # Basic datetime features
            processed_data[f"{col}_year"] = processed_data[col].dt.year
            processed_data[f"{col}_month"] = processed_data[col].dt.month
            processed_data[f"{col}_day"] = processed_data[col].dt.day
            processed_data[f"{col}_weekday"] = processed_data[col].dt.weekday
            processed_data[f"{col}_hour"] = processed_data[col].dt.hour
            processed_data[f"{col}_quarter"] = processed_data[col].dt.quarter
            
            # Cyclical features
            processed_data[f"{col}_month_sin"] = np.sin(2 * np.pi * processed_data[col].dt.month / 12)
            processed_data[f"{col}_month_cos"] = np.cos(2 * np.pi * processed_data[col].dt.month / 12)
            processed_data[f"{col}_day_sin"] = np.sin(2 * np.pi * processed_data[col].dt.day / 31)
            processed_data[f"{col}_day_cos"] = np.cos(2 * np.pi * processed_data[col].dt.day / 31)
        
        return processed_data
    
    @handle_errors("data")
    def validate_processed_data(self) -> Dict[str, Any]:
        """Validate processed data and generate quality report."""
        if self.data is None:
            raise ValidationError("No data loaded")
        
        validation_report = {
            'data_quality': self.validator.validate_dataframe(self.data),
            'processing_summary': self._get_processing_summary(),
            'data_drift': self._check_data_drift(),
            'recommendations': self._get_processing_recommendations()
        }
        
        return validation_report
    
    def _get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing operations."""
        if self.original_data is None:
            return {}
        
        return {
            'original_shape': self.original_data.shape,
            'processed_shape': self.data.shape,
            'columns_added': list(set(self.data.columns) - set(self.original_data.columns)),
            'columns_removed': list(set(self.original_data.columns) - set(self.data.columns)),
            'processing_steps': len(self.processing_log),
            'memory_reduction': (
                self.original_data.memory_usage(deep=True).sum() - 
                self.data.memory_usage(deep=True).sum()
            ) / 1024**2  # MB
        }
    
    def _check_data_drift(self) -> Dict[str, Any]:
        """Check for data drift between original and processed data."""
        if self.original_data is None:
            return {}
        
        drift_report = {}
        
        # Compare common columns
        common_cols = set(self.original_data.columns) & set(self.data.columns)
        
        for col in common_cols:
            if pd.api.types.is_numeric_dtype(self.original_data[col]) and pd.api.types.is_numeric_dtype(self.data[col]):
                # Compare distributions for numeric columns
                original_mean = self.original_data[col].mean()
                processed_mean = self.data[col].mean()
                mean_change = abs(original_mean - processed_mean) / abs(original_mean) if original_mean != 0 else 0
                
                drift_report[col] = {
                    'type': 'numeric',
                    'mean_change_percent': mean_change * 100,
                    'drift_detected': mean_change > 0.1  # 10% change threshold
                }
        
        return drift_report
    
    def _get_processing_recommendations(self) -> List[str]:
        """Get recommendations based on processing results."""
        recommendations = []
        
        if self.data is None:
            return recommendations
        
        # Check for high cardinality categorical variables
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if self.data[col].nunique() > 100:
                recommendations.append(f"Consider reducing cardinality of column '{col}'")
        
        # Check for highly skewed numeric variables
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            skewness = self.data[col].skew()
            if abs(skewness) > 2:
                recommendations.append(f"Consider transforming skewed column '{col}' (skewness: {skewness:.2f})")
        
        # Check for multicollinearity
        if len(numeric_cols) > 1:
            corr_matrix = self.data[numeric_cols].corr().abs()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.9:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                recommendations.append(f"High correlation detected between: {high_corr_pairs}")
        
        return recommendations
    
    @rate_limit()
    def get_processing_report(self) -> Dict[str, Any]:
        """Get comprehensive processing report."""
        return {
            'processing_log': self.processing_log,
            'data_info': {
                'current_shape': self.data.shape if self.data is not None else None,
                'original_shape': self.original_data.shape if self.original_data is not None else None,
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2 if self.data is not None else 0
            },
            'validation_report': self.validate_processed_data() if self.data is not None else None
        }
    
    def reset_processor(self):
        """Reset processor for new data."""
        self.data = None
        self.original_data = None
        self.processing_log = []
    
    def export_processed_data(self, filename: str, format: str = 'csv') -> str:
        """Export processed data to file."""
        if self.data is None:
            raise ValidationError("No data to export")
        
        if format == 'csv':
            filepath = f"data/{filename}.csv"
            self.data.to_csv(filepath, index=False)
        elif format == 'excel':
            filepath = f"data/{filename}.xlsx"
            self.data.to_excel(filepath, index=False)
        elif format == 'parquet':
            filepath = f"data/{filename}.parquet"
            self.data.to_parquet(filepath, index=False)
        else:
            raise ValidationError(f"Unsupported export format: {format}")
        
        self._log_processing(f"Data exported to {filepath}")
        return filepath


# Standalone functions for backward compatibility
def load_data_from_file(file_obj) -> pd.DataFrame:
    """Load data from uploaded file object."""
    try:
        if file_obj.name.endswith('.csv'):
            return pd.read_csv(file_obj)
        elif file_obj.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_obj)
        elif file_obj.name.endswith('.json'):
            return pd.read_json(file_obj)
        elif file_obj.name.endswith('.parquet'):
            return pd.read_parquet(file_obj)
        elif file_obj.name.endswith('.tsv'):
            return pd.read_csv(file_obj, sep='\t')
        else:
            raise ValueError(f"Unsupported file format: {file_obj.name}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def profile_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate data profile report."""
    try:
        processor = DataProcessor()
        processor.data = df
        processor.original_data = df.copy()
        
        # Basic info
        basic_info = {
            'rows': len(df),
            'columns': len(df.columns),
            'duplicates': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Column info
        column_info = {}
        for col in df.columns:
            column_info[col] = {
                'dtype': str(df[col].dtype),
                'non_null': df[col].count(),
                'null': df[col].isnull().sum(),
                'unique': df[col].nunique()
            }
        
        # Quality score
        from utils.helpers import calculate_data_quality_score
        quality_score = calculate_data_quality_score(df)
        
        return {
            'basic_info': basic_info,
            'column_info': column_info,
            'quality_score': quality_score
        }
    except Exception as e:
        print(f"Error profiling data: {e}")
        return None


def clean_data(df: pd.DataFrame, missing_strategy: str = 'mean',
               remove_duplicates: bool = True, outlier_strategy: str = 'none') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean data with specified strategies."""
    try:
        processor = DataProcessor()
        processor.data = df
        processor.original_data = df.copy()
        
        cleaning_config = {
            'handle_missing': missing_strategy != 'none',
            'missing_strategy': missing_strategy if missing_strategy != 'none' else 'auto',
            'handle_duplicates': remove_duplicates,
            'duplicate_strategy': 'drop' if remove_duplicates else 'none',
            'handle_outliers': outlier_strategy != 'none',
            'outlier_strategy': outlier_strategy if outlier_strategy != 'none' else 'iqr'
        }
        
        cleaned_df = processor.clean_data(cleaning_config)
        
        # Generate report
        report = {
            'actions': processor.processing_log,
            'original_shape': df.shape,
            'cleaned_shape': cleaned_df.shape,
            'columns_removed': list(set(df.columns) - set(cleaned_df.columns))
        }
        
        return cleaned_df, report
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return df, {'actions': [], 'error': str(e)}

