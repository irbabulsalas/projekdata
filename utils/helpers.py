"""
Helpers Module for AI Data Analysis Platform
Comprehensive utility functions for data analysis, ML, and general operations.
"""

import os
import re
import json
import pickle
import hashlib
import base64
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


# Data Type Detection and Conversion
class DataTypeDetector:
    """Advanced data type detection and conversion utilities."""
    
    @staticmethod
    def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
        """Detect and classify column types."""
        type_mapping = {}
        
        for col in df.columns:
            series = df[col].dropna()
            
            if series.empty:
                type_mapping[col] = 'empty'
                continue
            
            # Check for datetime
            if DataTypeDetector._is_datetime(series):
                type_mapping[col] = 'datetime'
            # Check for numeric
            elif DataTypeDetector._is_numeric(series):
                if DataTypeDetector._is_integer(series):
                    type_mapping[col] = 'integer'
                else:
                    type_mapping[col] = 'float'
            # Check for categorical
            elif DataTypeDetector._is_categorical(series):
                type_mapping[col] = 'categorical'
            # Check for text
            elif DataTypeDetector._is_text(series):
                type_mapping[col] = 'text'
            # Check for boolean
            elif DataTypeDetector._is_boolean(series):
                type_mapping[col] = 'boolean'
            else:
                type_mapping[col] = 'mixed'
        
        return type_mapping
    
    @staticmethod
    def _is_datetime(series: pd.Series) -> bool:
        """Check if series contains datetime data."""
        try:
            pd.to_datetime(series.head(100), errors='raise')
            return True
        except:
            return False
    
    @staticmethod
    def _is_numeric(series: pd.Series) -> bool:
        """Check if series contains numeric data."""
        try:
            pd.to_numeric(series.head(100), errors='raise')
            return True
        except:
            return False
    
    @staticmethod
    def _is_integer(series: pd.Series) -> bool:
        """Check if numeric series is integer."""
        try:
            numeric_series = pd.to_numeric(series.head(100), errors='raise')
            return (numeric_series % 1 == 0).all()
        except:
            return False
    
    @staticmethod
    def _is_categorical(series: pd.Series) -> bool:
        """Check if series is categorical."""
        unique_ratio = series.nunique() / len(series)
        return unique_ratio < 0.1 and series.nunique() <= 50
    
    @staticmethod
    def _is_text(series: pd.Series) -> bool:
        """Check if series contains text data."""
        if series.dtype == 'object':
            avg_length = series.astype(str).str.len().mean()
            return avg_length > 20
        return False
    
    @staticmethod
    def _is_boolean(series: pd.Series) -> bool:
        """Check if series contains boolean data."""
        unique_values = set(series.dropna().astype(str).str.lower())
        boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n'}
        return unique_values.issubset(boolean_values)


# File Operations
class FileHandler:
    """Advanced file handling utilities."""
    
    @staticmethod
    def save_file(data: Any, filename: str, file_type: str = 'pickle') -> str:
        """Save data to file with automatic directory creation."""
        os.makedirs('data', exist_ok=True)
        filepath = f"data/{filename}"
        
        if file_type == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        elif file_type == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, default=str)
        elif file_type == 'csv' and isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        elif file_type == 'excel' and isinstance(data, pd.DataFrame):
            data.to_excel(filepath, index=False)
        
        return filepath
    
    @staticmethod
    def load_file(filename: str, file_type: str = 'pickle') -> Any:
        """Load data from file."""
        filepath = f"data/{filename}"
        
        if file_type == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif file_type == 'json':
            with open(filepath, 'r') as f:
                return json.load(f)
        elif file_type == 'csv':
            return pd.read_csv(filepath)
        elif file_type == 'excel':
            return pd.read_excel(filepath)
        
        return None
    
    @staticmethod
    def get_file_hash(filepath: str) -> str:
        """Get MD5 hash of file."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def create_download_link(df: pd.DataFrame, filename: str, file_type: str = 'csv') -> str:
        """Create download link for DataFrame."""
        if file_type == 'csv':
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {file_type.upper()}</a>'
        elif file_type == 'excel':
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            b64 = base64.b64encode(output.getvalue()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download {file_type.upper()}</a>'
        
        return href


# Data Validation and Cleaning
class DataValidator:
    """Comprehensive data validation utilities."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate DataFrame and return quality report."""
        report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'quality_score': 0
        }
        
        # Calculate quality score
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        duplicate_ratio = df.duplicated().sum() / len(df)
        
        quality_score = (1 - (missing_cells / total_cells)) * (1 - duplicate_ratio)
        report['quality_score'] = round(quality_score, 3)
        
        return report
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, method: str = 'iqr') -> Dict[str, List[int]]:
        """Detect outliers in numeric columns."""
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = df[col].dropna()
            
            if method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (series < lower_bound) | (series > upper_bound)
            elif method == 'zscore':
                z_scores = np.abs((series - series.mean()) / series.std())
                outlier_mask = z_scores > 3
            
            outliers[col] = df[outlier_mask].index.tolist()
        
        return outliers
    
    @staticmethod
    def suggest_cleaning_steps(df: pd.DataFrame) -> List[str]:
        """Suggest data cleaning steps based on data quality."""
        suggestions = []
        
        # Check missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            suggestions.append(f"Handle missing values in columns: {missing_cols}")
        
        # Check duplicates
        if df.duplicated().any():
            suggestions.append("Remove duplicate rows")
        
        # Check data types
        type_detector = DataTypeDetector()
        detected_types = type_detector.detect_column_types(df)
        for col, detected_type in detected_types.items():
            if detected_type == 'mixed':
                suggestions.append(f"Standardize data type for column: {col}")
        
        # Check outliers
        outliers = DataValidator.detect_outliers(df)
        for col, outlier_indices in outliers.items():
            if outlier_indices:
                suggestions.append(f"Review outliers in column: {col} ({len(outlier_indices)} outliers)")
        
        return suggestions


# Text Processing Utilities
class TextProcessor:
    """Advanced text processing utilities."""
    
    @staticmethod
    def clean_text(text: str, remove_special_chars: bool = True, 
                   remove_numbers: bool = False, lowercase: bool = True) -> str:
        """Clean text data."""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        if lowercase:
            text = text.lower()
        
        if remove_special_chars:
            text = re.sub(r'[^\w\s]', '', text)
        
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> List[str]:
        """Extract keywords from text using simple frequency analysis."""
        from collections import Counter
        
        # Simple keyword extraction (can be enhanced with NLP libraries)
        words = TextProcessor.clean_text(text).split()
        word_freq = Counter(words)
        
        # Remove common stop words (basic list)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        
        # Filter stop words and short words
        keywords = [(word, freq) for word, freq in word_freq.items() 
                   if word not in stop_words and len(word) > 2]
        
        # Sort by frequency and return top N
        keywords.sort(key=lambda x: x[1], reverse=True)
        return [word for word, freq in keywords[:top_n]]
    
    @staticmethod
    def calculate_readability(text: str) -> Dict[str, float]:
        """Calculate basic readability scores."""
        if not text:
            return {'flesch_score': 0, 'avg_sentence_length': 0, 'avg_word_length': 0}
        
        sentences = text.split('.')
        words = text.split()
        
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Simplified Flesch Reading Ease score
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
        
        return {
            'flesch_score': round(flesch_score, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_word_length': round(avg_word_length, 2)
        }


# Visualization Utilities
class VisualizationHelper:
    """Advanced visualization utilities."""
    
    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
        """Create interactive correlation heatmap."""
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Features",
            yaxis_title="Features",
            width=600,
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_distribution_plot(df: pd.DataFrame, column: str) -> go.Figure:
        """Create comprehensive distribution plot."""
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=df[column],
            name='Distribution',
            nbinsx=30,
            opacity=0.7
        ))
        
        # Add box plot
        fig.add_trace(go.Box(
            y=df[column],
            name='Box Plot',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f'Distribution of {column}',
            xaxis_title=column,
            yaxis_title='Frequency',
            yaxis2=dict(
                title='Value',
                overlaying='y',
                side='right'
            ),
            barmode='overlay'
        )
        
        return fig
    
    @staticmethod
    def create_missing_values_plot(df: pd.DataFrame) -> go.Figure:
        """Create missing values visualization."""
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
        
        fig = go.Figure(data=go.Bar(
            x=missing_data.values,
            y=missing_data.index,
            orientation='h',
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title='Missing Values by Column',
            xaxis_title='Number of Missing Values',
            yaxis_title='Columns',
            height=max(400, len(missing_data) * 30)
        )
        
        return fig


# Performance Utilities
class PerformanceMonitor:
    """Performance monitoring utilities."""
    
    @staticmethod
    def measure_execution_time(func: Callable) -> Tuple[Any, float]:
        """Measure execution time of a function."""
        start_time = datetime.now()
        result = func()
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        return result, execution_time
    
    @staticmethod
    def memory_usage() -> Dict[str, float]:
        """Get current memory usage."""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / (1024 * 1024),  # MB
            'vms': memory_info.vms / (1024 * 1024),  # MB
            'percent': process.memory_percent()
        }


# Streamlit Utilities
class StreamlitHelper:
    """Streamlit-specific helper utilities."""
    
    @staticmethod
    def display_dataframe_info(df: pd.DataFrame, title: str = "Dataset Information"):
        """Display comprehensive DataFrame information."""
        st.subheader(title)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Rows", df.shape[0])
        
        with col2:
            st.metric("📋 Columns", df.shape[1])
        
        with col3:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.metric("🔍 Missing %", f"{missing_pct:.2f}%")
        
        # Data types
        st.write("**Data Types:**")
        type_counts = df.dtypes.value_counts()
        st.write(type_counts)
        
        # Sample data
        st.write("**Sample Data:**")
        st.dataframe(df.head())
    
    @staticmethod
    def create_expander(title: str, content_func: Callable, expanded: bool = False):
        """Create expander with dynamic content."""
        with st.expander(title, expanded=expanded):
            content_func()
    
    @staticmethod
    def display_progress_bar(progress: float, message: str = "Processing..."):
        """Display progress bar with message."""
        st.progress(progress)
        st.write(message)
    
    @staticmethod
    def create_sidebar_filters(df: pd.DataFrame) -> Dict[str, Any]:
        """Create interactive sidebar filters."""
        st.sidebar.subheader("🔍 Filters")
        
        filters = {}
        
        # Column selection
        selected_columns = st.sidebar.multiselect(
            "Select Columns",
            df.columns.tolist(),
            default=df.columns.tolist()
        )
        filters['columns'] = selected_columns
        
        # Row limit
        row_limit = st.sidebar.slider(
            "Row Limit",
            min_value=100,
            max_value=len(df),
            value=min(1000, len(df))
        )
        filters['row_limit'] = row_limit
        
        # Missing value threshold
        missing_threshold = st.sidebar.slider(
            "Missing Value Threshold (%)",
            min_value=0,
            max_value=100,
            value=50
        )
        filters['missing_threshold'] = missing_threshold
        
        return filters


# Configuration Management
class ConfigManager:
    """Configuration management utilities."""
    
    @staticmethod
    def load_config(config_file: str = 'config.json') -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_file: str = 'config.json'):
        """Save configuration to file."""
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def get_env_var(key: str, default: Any = None) -> Any:
        """Get environment variable with default value."""
        return os.getenv(key, default)


# Cache Management
class CacheManager:
    """Cache management utilities."""
    
    @staticmethod
    def cache_key(*args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @staticmethod
    def clear_cache():
        """Clear Streamlit cache."""
        if hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()


# Export Utilities
class ExportHelper:
    """Export utilities for various formats."""
    
    @staticmethod
    def export_to_pdf(content: str, filename: str = "report.pdf") -> bytes:
        """Export content to PDF."""
        from fpdf import FPDF
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, content)
        
        return pdf.output(dest='S').encode('latin-1')
    
    @staticmethod
    def create_data_report(df: pd.DataFrame) -> str:
        """Create comprehensive data report."""
        report = []
        report.append("DATA ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Basic info
        report.append("BASIC INFORMATION")
        report.append("-" * 20)
        report.append(f"Shape: {df.shape}")
        report.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        report.append("")
        
        # Data types
        report.append("DATA TYPES")
        report.append("-" * 20)
        for dtype, count in df.dtypes.value_counts().items():
            report.append(f"{dtype}: {count} columns")
        report.append("")
        
        # Missing values
        report.append("MISSING VALUES")
        report.append("-" * 20)
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            for col, count in missing_data.items():
                pct = (count / len(df)) * 100
                report.append(f"{col}: {count} ({pct:.2f}%)")
        else:
            report.append("No missing values found")
        report.append("")
        
        # Statistical summary
        report.append("STATISTICAL SUMMARY")
        report.append("-" * 20)
        report.append(df.describe().to_string())
        
        return "\n".join(report)


# Utility Functions
def format_number(num: float, precision: int = 2) -> str:
    """Format number with appropriate units."""
    if num >= 1e9:
        return f"{num/1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """Safe division with default value."""
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_directory_if_not_exists(directory: str):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def is_valid_email(email: str) -> bool:
    """Check if email is valid."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length."""
    return text[:max_length] + "..." if len(text) > max_length else text


# Streamlit Session State Management
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    if 'current_page' not in st.session_state:
        st.session_state.current_page = '📈 Overview'


# Data Analysis Helper Functions
def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric columns from DataFrame."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Get list of categorical columns from DataFrame."""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def calculate_data_quality_score(df: pd.DataFrame) -> float:
    """Calculate data quality score (0-100)."""
    try:
        if df.empty:
            return 0.0
        
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        # Base score
        score = 100.0
        
        # Deduct for missing values
        missing_penalty = (missing_cells / total_cells) * 30
        score -= missing_penalty
        
        # Deduct for duplicates
        duplicate_penalty = (duplicate_rows / len(df)) * 20
        score -= duplicate_penalty
        
        # Bonus for consistent data types
        type_consistency = 0
        for col in df.columns:
            if df[col].dtype != 'object':
                type_consistency += 1
        type_bonus = (type_consistency / len(df.columns)) * 10
        score += type_bonus
        
        return max(0.0, min(100.0, score))
    except Exception as e:
        print(f"Error calculating quality score: {e}")
        return 0.0


# Validation Functions
def validate_dataframe(df: pd.DataFrame, min_rows: int = 0) -> bool:
    """Validate DataFrame for analysis."""
    if df is None or df.empty:
        return False
    if min_rows > 0 and len(df) < min_rows:
        return False
    return True


# Error Handling Wrapper
def safe_execute(operation_name: str, func: Callable, *args, **kwargs):
    """Safely execute function with error handling."""
    try:
        result = func(*args, **kwargs)
        return result
    except Exception as e:
        st.error(f"❌ Error in {operation_name}: {str(e)}")
        return None