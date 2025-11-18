"""
Smart Data Pipeline Module for AI Data Analysis Platform
Intelligent automated data analysis pipeline with adaptive processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from utils.helpers import DataTypeDetector, DataValidator, PerformanceMonitor
except ImportError:
    class DataTypeDetector:
        @staticmethod
        def detect_column_types(df):
            return {col: str(df[col].dtype) for col in df.columns}
    
    class DataValidator:
        @staticmethod
        def validate_dataframe(df):
            return True
    
    class PerformanceMonitor:
        @staticmethod
        def measure_execution_time(func):
            return func(), 0

try:
    from utils.error_handler import handle_errors, DataProcessingError, ValidationError
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
    
    class DataProcessingError(Exception):
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


class SmartDataPipeline:
    """Intelligent data analysis pipeline with adaptive processing."""
    
    def __init__(self):
        """Initialize the smart pipeline."""
        self.data = None
        self.metadata = {}
        self.pipeline_steps = []
        self.results = {}
        self.type_detector = DataTypeDetector()
        self.validator = DataValidator()
        self.performance_monitor = PerformanceMonitor()
    
    @handle_errors("data")
    def load_data(self, data_source: Union[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """Load data from various sources with automatic format detection."""
        if isinstance(data_source, str):
            # Load from file path
            if data_source.endswith('.csv'):
                self.data = pd.read_csv(data_source, **kwargs)
            elif data_source.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(data_source, **kwargs)
            elif data_source.endswith('.json'):
                self.data = pd.read_json(data_source, **kwargs)
            elif data_source.endswith('.parquet'):
                self.data = pd.read_parquet(data_source, **kwargs)
            else:
                raise ValidationError(f"Unsupported file format: {data_source}")
        elif isinstance(data_source, pd.DataFrame):
            self.data = data_source.copy()
        else:
            raise ValidationError("Data source must be file path or DataFrame")
        
        # Initialize metadata
        self._initialize_metadata()
        
        return self.data
    
    def _initialize_metadata(self):
        """Initialize metadata for the loaded data."""
        self.metadata = {
            'load_time': datetime.now(),
            'original_shape': self.data.shape,
            'column_types': self.type_detector.detect_column_types(self.data),
            'validation_report': self.validator.validate_dataframe(self.data)
        }
    
    @handle_errors("data")
    def auto_detect_data_quality(self) -> Dict[str, Any]:
        """Automatically detect data quality issues."""
        if self.data is None:
            raise ValidationError("No data loaded")
        
        quality_report = {
            'overall_score': 0,
            'issues': [],
            'recommendations': [],
            'statistics': {}
        }
        
        # Missing values analysis
        missing_analysis = self._analyze_missing_values()
        quality_report['statistics']['missing_values'] = missing_analysis
        
        # Duplicate analysis
        duplicate_analysis = self._analyze_duplicates()
        quality_report['statistics']['duplicates'] = duplicate_analysis
        
        # Data type consistency
        type_analysis = self._analyze_data_types()
        quality_report['statistics']['data_types'] = type_analysis
        
        # Outlier detection
        outlier_analysis = self._analyze_outliers()
        quality_report['statistics']['outliers'] = outlier_analysis
        
        # Calculate overall quality score
        quality_report['overall_score'] = self._calculate_quality_score(quality_report['statistics'])
        
        # Generate recommendations
        quality_report['recommendations'] = self._generate_recommendations(quality_report['statistics'])
        
        self.results['data_quality'] = quality_report
        return quality_report
    
    def _analyze_missing_values(self) -> Dict[str, Any]:
        """Analyze missing values patterns."""
        missing_data = self.data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if missing_data.empty:
            return {'total_missing': 0, 'columns_with_missing': [], 'patterns': {}}
        
        total_missing = missing_data.sum()
        missing_pct = (missing_data / len(self.data)) * 100
        
        # Analyze patterns
        patterns = {}
        for col in missing_data.index:
            col_missing = self.data[col].isnull()
            patterns[col] = {
                'count': missing_data[col],
                'percentage': missing_pct[col],
                'consecutive_blocks': self._find_consecutive_missing_blocks(col_missing)
            }
        
        return {
            'total_missing': total_missing,
            'columns_with_missing': missing_data.index.tolist(),
            'patterns': patterns
        }
    
    def _find_consecutive_missing_blocks(self, missing_series: pd.Series) -> List[Dict]:
        """Find consecutive blocks of missing values."""
        blocks = []
        current_block = None
        
        for i, is_missing in enumerate(missing_series):
            if is_missing:
                if current_block is None:
                    current_block = {'start': i, 'end': i}
                else:
                    current_block['end'] = i
            else:
                if current_block is not None:
                    blocks.append(current_block)
                    current_block = None
        
        if current_block is not None:
            blocks.append(current_block)
        
        return blocks
    
    def _analyze_duplicates(self) -> Dict[str, Any]:
        """Analyze duplicate records."""
        duplicate_rows = self.data.duplicated()
        total_duplicates = duplicate_rows.sum()
        
        if total_duplicates == 0:
            return {'total_duplicates': 0, 'duplicate_percentage': 0}
        
        duplicate_percentage = (total_duplicates / len(self.data)) * 100
        
        # Find duplicate groups
        duplicate_groups = self.data[duplicate_rows].groupby(self.data.columns.tolist()).size()
        
        return {
            'total_duplicates': total_duplicates,
            'duplicate_percentage': duplicate_percentage,
            'duplicate_groups': duplicate_groups.to_dict()
        }
    
    def _analyze_data_types(self) -> Dict[str, Any]:
        """Analyze data type consistency and issues."""
        type_issues = {}
        
        for col in self.data.columns:
            col_issues = []
            
            # Check for mixed types in object columns
            if self.data[col].dtype == 'object':
                sample_types = set(type(x).__name__ for x in self.data[col].dropna().head(100))
                if len(sample_types) > 1:
                    col_issues.append(f"Mixed types detected: {sample_types}")
            
            # Check for numeric columns with many unique values (might be categorical)
            if pd.api.types.is_numeric_dtype(self.data[col]):
                unique_ratio = self.data[col].nunique() / len(self.data)
                if unique_ratio < 0.05 and self.data[col].nunique() > 2:
                    col_issues.append("Might be categorical disguised as numeric")
            
            if col_issues:
                type_issues[col] = col_issues
        
        return {
            'type_distribution': self.data.dtypes.value_counts().to_dict(),
            'issues': type_issues
        }
    
    def _analyze_outliers(self) -> Dict[str, Any]:
        """Analyze outliers in numeric columns."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        outlier_analysis = {}
        
        for col in numeric_cols:
            series = self.data[col].dropna()
            
            if len(series) < 10:  # Skip small samples
                continue
            
            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            if len(outliers) > 0:
                outlier_analysis[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(series)) * 100,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound},
                    'values': outliers.tolist()[:10]  # First 10 outliers
                }
        
        return outlier_analysis
    
    def _calculate_quality_score(self, statistics: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        score = 100.0
        
        # Deduct for missing values
        missing_stats = statistics.get('missing_values', {})
        if missing_stats.get('total_missing', 0) > 0:
            missing_penalty = min(30, (missing_stats['total_missing'] / (len(self.data) * len(self.data.columns))) * 100)
            score -= missing_penalty
        
        # Deduct for duplicates
        duplicate_stats = statistics.get('duplicates', {})
        if duplicate_stats.get('total_duplicates', 0) > 0:
            duplicate_penalty = min(20, duplicate_stats['duplicate_percentage'])
            score -= duplicate_penalty
        
        # Deduct for type issues
        type_stats = statistics.get('data_types', {})
        if type_stats.get('issues'):
            type_penalty = min(15, len(type_stats['issues']) * 3)
            score -= type_penalty
        
        # Deduct for outliers
        outlier_stats = statistics.get('outliers', {})
        if outlier_stats:
            outlier_penalty = min(10, len(outlier_stats) * 2)
            score -= outlier_penalty
        
        return max(0, round(score, 2))
    
    def _generate_recommendations(self, statistics: Dict[str, Any]) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []
        
        # Missing values recommendations
        missing_stats = statistics.get('missing_values', {})
        if missing_stats.get('total_missing', 0) > 0:
            recommendations.append("Handle missing values using imputation or removal strategies")
            
            # Specific recommendations for high missing columns
            for col, pattern in missing_stats.get('patterns', {}).items():
                if pattern['percentage'] > 50:
                    recommendations.append(f"Consider dropping column '{col}' (>{pattern['percentage']:.1f}% missing)")
                elif pattern['percentage'] > 20:
                    recommendations.append(f"Apply imputation for column '{col}' ({pattern['percentage']:.1f}% missing)")
        
        # Duplicates recommendations
        duplicate_stats = statistics.get('duplicates', {})
        if duplicate_stats.get('total_duplicates', 0) > 0:
            recommendations.append("Remove duplicate records to improve data quality")
        
        # Data type recommendations
        type_stats = statistics.get('data_types', {})
        if type_stats.get('issues'):
            recommendations.append("Review and standardize data types for consistency")
        
        # Outlier recommendations
        outlier_stats = statistics.get('outliers', {})
        if outlier_stats:
            recommendations.append("Investigate and handle outliers appropriately")
        
        return recommendations
    
    @handle_errors("data")
    def auto_preprocess_data(self, strategy: str = 'conservative') -> pd.DataFrame:
        """Automatically preprocess data based on detected issues."""
        if self.data is None:
            raise ValidationError("No data loaded")
        
        processed_data = self.data.copy()
        preprocessing_log = []
        
        # Handle missing values
        processed_data, missing_log = self._auto_handle_missing_values(processed_data, strategy)
        preprocessing_log.extend(missing_log)
        
        # Handle duplicates
        processed_data, duplicate_log = self._auto_handle_duplicates(processed_data)
        preprocessing_log.extend(duplicate_log)
        
        # Handle data types
        processed_data, type_log = self._auto_handle_data_types(processed_data)
        preprocessing_log.extend(type_log)
        
        # Handle outliers
        processed_data, outlier_log = self._auto_handle_outliers(processed_data, strategy)
        preprocessing_log.extend(outlier_log)
        
        self.results['preprocessing'] = {
            'processed_data': processed_data,
            'log': preprocessing_log,
            'strategy': strategy
        }
        
        return processed_data
    
    def _auto_handle_missing_values(self, data: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, List[str]]:
        """Automatically handle missing values."""
        log = []
        processed_data = data.copy()
        
        for col in processed_data.columns:
            missing_count = processed_data[col].isnull().sum()
            if missing_count == 0:
                continue
            
            missing_pct = (missing_count / len(processed_data)) * 100
            
            if missing_pct > 50:
                # Drop column with too many missing values
                processed_data = processed_data.drop(columns=[col])
                log.append(f"Dropped column '{col}' ({missing_pct:.1f}% missing)")
            elif missing_pct > 20:
                # Impute based on data type
                if pd.api.types.is_numeric_dtype(processed_data[col]):
                    processed_data[col].fillna(processed_data[col].median(), inplace=True)
                    log.append(f"Imputed median for column '{col}' ({missing_pct:.1f}% missing)")
                else:
                    processed_data[col].fillna(processed_data[col].mode().iloc[0] if not processed_data[col].mode().empty else 'Unknown', inplace=True)
                    log.append(f"Imputed mode for column '{col}' ({missing_pct:.1f}% missing)")
            else:
                # Simple imputation
                if pd.api.types.is_numeric_dtype(processed_data[col]):
                    processed_data[col].fillna(processed_data[col].mean(), inplace=True)
                else:
                    processed_data[col].fillna(processed_data[col].mode().iloc[0] if not processed_data[col].mode().empty else 'Unknown', inplace=True)
        
        return processed_data, log
    
    def _auto_handle_duplicates(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Automatically handle duplicate records."""
        log = []
        processed_data = data.copy()
        
        duplicate_count = processed_data.duplicated().sum()
        if duplicate_count > 0:
            processed_data = processed_data.drop_duplicates()
            log.append(f"Removed {duplicate_count} duplicate records")
        
        return processed_data, log
    
    def _auto_handle_data_types(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Automatically handle data type issues."""
        log = []
        processed_data = data.copy()
        
        # Convert object columns to appropriate types
        for col in processed_data.select_dtypes(include=['object']).columns:
            # Try to convert to datetime
            try:
                processed_data[col] = pd.to_datetime(processed_data[col], errors='raise')
                log.append(f"Converted column '{col}' to datetime")
                continue
            except:
                pass
            
            # Try to convert to numeric
            try:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='raise')
                log.append(f"Converted column '{col}' to numeric")
                continue
            except:
                pass
            
            # Convert to category if low cardinality
            unique_ratio = processed_data[col].nunique() / len(processed_data)
            if unique_ratio < 0.1 and processed_data[col].nunique() <= 50:
                processed_data[col] = processed_data[col].astype('category')
                log.append(f"Converted column '{col}' to category")
        
        return processed_data, log
    
    def _auto_handle_outliers(self, data: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, List[str]]:
        """Automatically handle outliers."""
        log = []
        processed_data = data.copy()
        
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = processed_data[col].dropna()
            
            if len(series) < 10:
                continue
            
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (processed_data[col] < lower_bound) | (processed_data[col] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                if strategy == 'conservative':
                    # Cap outliers
                    processed_data[col] = processed_data[col].clip(lower=lower_bound, upper=upper_bound)
                    log.append(f"Capped {outlier_count} outliers in column '{col}'")
                elif strategy == 'aggressive':
                    # Remove outliers
                    processed_data = processed_data[~outliers]
                    log.append(f"Removed {outlier_count} outliers from column '{col}'")
        
        return processed_data, log
    
    @handle_errors("data")
    def generate_eda_summary(self) -> Dict[str, Any]:
        """Generate comprehensive exploratory data analysis summary."""
        if self.data is None:
            raise ValidationError("No data loaded")
        
        summary = {
            'dataset_info': self._get_dataset_info(),
            'statistical_summary': self._get_statistical_summary(),
            'correlation_analysis': self._get_correlation_analysis(),
            'distribution_analysis': self._get_distribution_analysis(),
            'feature_analysis': self._get_feature_analysis()
        }
        
        self.results['eda_summary'] = summary
        return summary
    
    def _get_dataset_info(self) -> Dict[str, Any]:
        """Get basic dataset information."""
        return {
            'shape': self.data.shape,
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
            'column_types': self.data.dtypes.value_counts().to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicate_rows': self.data.duplicated().sum()
        }
    
    def _get_statistical_summary(self) -> Dict[str, Any]:
        """Get statistical summary of the data."""
        numeric_summary = self.data.describe().to_dict()
        
        categorical_summary = {}
        for col in self.data.select_dtypes(include=['object', 'category']).columns:
            categorical_summary[col] = {
                'unique_count': self.data[col].nunique(),
                'most_frequent': self.data[col].mode().iloc[0] if not self.data[col].mode().empty else None,
                'frequency': self.data[col].value_counts().iloc[0] if not self.data[col].empty else 0
            }
        
        return {
            'numeric': numeric_summary,
            'categorical': categorical_summary
        }
    
    def _get_correlation_analysis(self) -> Dict[str, Any]:
        """Get correlation analysis for numeric columns."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {'message': 'No numeric columns for correlation analysis'}
        
        correlation_matrix = numeric_data.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations
        }
    
    def _get_distribution_analysis(self) -> Dict[str, Any]:
        """Get distribution analysis for numeric columns."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {'message': 'No numeric columns for distribution analysis'}
        
        distribution_info = {}
        for col in numeric_data.columns:
            series = numeric_data[col].dropna()
            
            if len(series) == 0:
                continue
            
            distribution_info[col] = {
                'skewness': series.skew(),
                'kurtosis': series.kurtosis(),
                'is_normal': abs(series.skew()) < 0.5 and abs(series.kurtosis()) < 0.5,
                'distribution_type': self._classify_distribution(series)
            }
        
        return distribution_info
    
    def _classify_distribution(self, series: pd.Series) -> str:
        """Classify the distribution type."""
        skewness = series.skew()
        kurtosis = series.kurtosis()
        
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return 'normal'
        elif skewness > 1:
            return 'right_skewed'
        elif skewness < -1:
            return 'left_skewed'
        elif kurtosis > 1:
            return 'heavy_tailed'
        elif kurtosis < -1:
            return 'light_tailed'
        else:
            return 'unknown'
    
    def _get_feature_analysis(self) -> Dict[str, Any]:
        """Get detailed feature analysis."""
        feature_analysis = {}
        
        for col in self.data.columns:
            col_info = {
                'data_type': str(self.data[col].dtype),
                'unique_count': self.data[col].nunique(),
                'missing_count': self.data[col].isnull().sum(),
                'missing_percentage': (self.data[col].isnull().sum() / len(self.data)) * 100
            }
            
            # Add type-specific analysis
            if pd.api.types.is_numeric_dtype(self.data[col]):
                col_info.update({
                    'min': self.data[col].min(),
                    'max': self.data[col].max(),
                    'mean': self.data[col].mean(),
                    'std': self.data[col].std()
                })
            elif pd.api.types.is_datetime64_any_dtype(self.data[col]):
                col_info.update({
                    'min_date': self.data[col].min(),
                    'max_date': self.data[col].max(),
                    'date_range_days': (self.data[col].max() - self.data[col].min()).days
                })
            else:
                # Categorical/text analysis
                value_counts = self.data[col].value_counts()
                col_info.update({
                    'most_common': value_counts.index[0] if not value_counts.empty else None,
                    'most_common_count': value_counts.iloc[0] if not value_counts.empty else 0,
                    'avg_length': self.data[col].astype(str).str.len().mean() if self.data[col].dtype == 'object' else None
                })
            
            feature_analysis[col] = col_info
        
        return feature_analysis
    
    @rate_limit()
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get complete pipeline summary with all results."""
        return {
            'metadata': self.metadata,
            'results': self.results,
            'pipeline_steps': self.pipeline_steps,
            'performance_metrics': self._get_performance_metrics()
        }
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the pipeline."""
        return {
            'data_load_time': self.metadata.get('load_time'),
            'processing_time': datetime.now(),
            'memory_usage': self.data.memory_usage(deep=True).sum() / 1024**2 if self.data is not None else 0
        }
    
    def reset_pipeline(self):
        """Reset the pipeline for new analysis."""
        self.data = None
        self.metadata = {}
        self.pipeline_steps = []
        self.results = {}