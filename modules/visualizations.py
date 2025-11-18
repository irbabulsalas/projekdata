"""
Advanced Visualizations Module for AI Data Analysis Platform
Comprehensive interactive visualization utilities with Plotly and advanced charts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Visualization Libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import networkx as nx

# Statistical visualization
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.offline as pyo

try:
    from utils.helpers import DataTypeDetector
except ImportError:
    class DataTypeDetector:
        @staticmethod
        def detect_column_types(df):
            return {col: str(df[col].dtype) for col in df.columns}

try:
    from utils.error_handler import handle_errors, ValidationError
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
    
    class ValidationError(Exception):
        pass

try:
    from utils.rate_limiter import rate_limit
except ImportError:
    def rate_limit():
        def decorator(func):
            return func
        return decorator


class AdvancedVisualizer:
    """Advanced visualization system with interactive charts and analytics."""
    
    def __init__(self):
        """Initialize advanced visualizer."""
        self.data = None
        self.type_detector = DataTypeDetector()
        self.color_palette = px.colors.qualitative.Set3
        self.theme = 'plotly_white'
    
    @handle_errors("data")
    def set_data(self, data: pd.DataFrame):
        """Set data for visualization."""
        if data is None or data.empty:
            raise ValidationError("No data provided for visualization")
        
        self.data = data.copy()
    
    @handle_errors("data")
    def create_data_overview_dashboard(self) -> Dict[str, go.Figure]:
        """Create comprehensive data overview dashboard."""
        if self.data is None:
            raise ValidationError("No data set for visualization")
        
        dashboard = {}
        
        # Basic info cards
        dashboard['info_cards'] = self._create_info_cards()
        
        # Data types distribution
        dashboard['data_types'] = self._create_data_types_chart()
        
        # Missing values heatmap
        dashboard['missing_values'] = self._create_missing_values_heatmap()
        
        # Correlation matrix
        dashboard['correlation'] = self._create_correlation_matrix()
        
        # Data quality score
        dashboard['quality_score'] = self._create_quality_score_gauge()
        
        return dashboard
    
    def _create_info_cards(self) -> go.Figure:
        """Create information cards."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('📊 Dataset Shape', '🔍 Missing Values', '📋 Columns', '💾 Memory Usage'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Dataset shape
        fig.add_trace(
            go.Indicator(
                mode="number+gauge+title",
                value=self.data.shape[0],
                title={"text": f"Rows<br><span style='font-size:0.8em;color:gray'>Columns: {self.data.shape[1]}</span>"},
                gauge={'axis': {'range': [None, max(1000, self.data.shape[0])]}},
                domain={'row': 0, 'column': 0}
            ),
            row=1, col=1
        )
        
        # Missing values
        missing_pct = (self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1])) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+title",
                value=missing_pct,
                title={"text": "Missing %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 10], 'color': "lightgray"},
                                {'range': [10, 30], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 50}},
                domain={'row': 0, 'column': 1}
            ),
            row=1, col=2
        )
        
        # Column types
        type_counts = self.data.dtypes.value_counts()
        fig.add_trace(
            go.Pie(
                labels=type_counts.index,
                values=type_counts.values,
                name="Column Types"
            ),
            row=2, col=1
        )
        
        # Memory usage
        memory_mb = self.data.memory_usage(deep=True).sum() / 1024**2
        fig.add_trace(
            go.Indicator(
                mode="number+title",
                value=memory_mb,
                title={"text": "Memory (MB)"},
                domain={'row': 1, 'column': 1}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Dataset Overview Dashboard"
        )
        
        return fig
    
    def _create_data_types_chart(self) -> go.Figure:
        """Create data types distribution chart."""
        type_counts = self.data.dtypes.value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=type_counts.index,
                y=type_counts.values,
                marker_color=self.color_palette[:len(type_counts)],
                text=type_counts.values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Data Types Distribution",
            xaxis_title="Data Type",
            yaxis_title="Count",
            template=self.theme
        )
        
        return fig
    
    def _create_missing_values_heatmap(self) -> go.Figure:
        """Create missing values heatmap."""
        missing_data = self.data.isnull()
        
        if not missing_data.any().any():
            # No missing values
            fig = go.Figure()
            fig.add_annotation(
                text="No missing values found! 🎉",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                font=dict(size=20, color="green")
            )
            return fig
        
        fig = go.Figure(data=go.Heatmap(
            z=missing_data.astype(int),
            x=self.data.columns,
            y=list(range(len(self.data))),
            colorscale='Reds',
            showscale=True,
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Missing Values Heatmap",
            xaxis_title="Columns",
            yaxis_title="Rows",
            height=400
        )
        
        return fig
    
    def _create_correlation_matrix(self) -> go.Figure:
        """Create correlation matrix for numeric columns."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No numeric columns for correlation analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                font=dict(size=16)
            )
            return fig
        
        corr_matrix = numeric_data.corr()
        
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
            title="Correlation Matrix",
            width=600,
            height=600,
            template=self.theme
        )
        
        return fig
    
    def _create_quality_score_gauge(self) -> go.Figure:
        """Create data quality score gauge."""
        # Calculate quality score
        total_cells = self.data.shape[0] * self.data.shape[1]
        missing_cells = self.data.isnull().sum().sum()
        duplicate_rows = self.data.duplicated().sum()
        
        quality_score = (1 - (missing_cells / total_cells)) * (1 - (duplicate_rows / len(self.data)))
        quality_score = max(0, min(100, quality_score * 100))
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=quality_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Data Quality Score"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400, template=self.theme)
        
        return fig
    
    @handle_errors("data")
    def create_eda_visualizations(self, target_column: str = None) -> Dict[str, go.Figure]:
        """Create comprehensive EDA visualizations."""
        if self.data is None:
            raise ValidationError("No data set for visualization")
        
        visualizations = {}
        
        # Distribution plots for numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            visualizations['distributions'] = self._create_distribution_plots(numeric_cols)
        
        # Categorical plots
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            visualizations['categorical'] = self._create_categorical_plots(categorical_cols)
        
        # Target analysis
        if target_column and target_column in self.data.columns:
            visualizations['target_analysis'] = self._create_target_analysis(target_column)
        
        # Relationship plots
        if len(numeric_cols) > 1:
            visualizations['relationships'] = self._create_relationship_plots(numeric_cols)
        
        # Time series plots
        datetime_cols = self.data.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            visualizations['time_series'] = self._create_time_series_plots(datetime_cols)
        
        return visualizations
    
    def _create_distribution_plots(self, numeric_cols: List[str]) -> go.Figure:
        """Create distribution plots for numeric columns."""
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=numeric_cols[:n_rows*n_cols],
            specs=[[{"secondary_y": True}] * n_cols] * n_rows
        )
        
        for i, col in enumerate(numeric_cols[:n_rows*n_cols]):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            
            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=self.data[col].dropna(),
                    name=f'{col} Distribution',
                    opacity=0.7,
                    nbinsx=30
                ),
                row=row, col=col_idx
            )
            
            # Box plot
            fig.add_trace(
                go.Box(
                    y=self.data[col].dropna(),
                    name=f'{col} Box Plot',
                    boxpoints='outliers'
                ),
                row=row, col=col_idx, secondary_y=True
            )
        
        fig.update_layout(
            height=300 * n_rows,
            title_text="Numeric Distributions",
            showlegend=False,
            template=self.theme
        )
        
        return fig
    
    def _create_categorical_plots(self, categorical_cols: List[str]) -> go.Figure:
        """Create plots for categorical columns."""
        n_cols = min(3, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=categorical_cols[:n_rows*n_cols],
            specs=[[{"type": "domain"}] * n_cols] * n_rows
        )
        
        for i, col in enumerate(categorical_cols[:n_rows*n_cols]):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            
            value_counts = self.data[col].value_counts().head(10)  # Top 10 categories
            
            fig.add_trace(
                go.Pie(
                    labels=value_counts.index,
                    values=value_counts.values,
                    name=col
                ),
                row=row, col=col_idx
            )
        
        fig.update_layout(
            height=300 * n_rows,
            title_text="Categorical Distributions",
            template=self.theme
        )
        
        return fig
    
    def _create_target_analysis(self, target_column: str) -> go.Figure:
        """Create target variable analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribution', 'Box Plot', 'Value Counts', 'Missing Pattern'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Distribution
        if pd.api.types.is_numeric_dtype(self.data[target_column]):
            fig.add_trace(
                go.Histogram(
                    x=self.data[target_column].dropna(),
                    name='Distribution',
                    nbinsx=30
                ),
                row=1, col=1
            )
        else:
            value_counts = self.data[target_column].value_counts()
            fig.add_trace(
                go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    name='Value Counts'
                ),
                row=1, col=1
            )
        
        # Box plot (for numeric)
        if pd.api.types.is_numeric_dtype(self.data[target_column]):
            fig.add_trace(
                go.Box(
                    y=self.data[target_column].dropna(),
                    name='Box Plot'
                ),
                row=1, col=2
            )
        
        # Value counts
        value_counts = self.data[target_column].value_counts().head(10)
        fig.add_trace(
            go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                name='Top Values'
            ),
            row=2, col=1
        )
        
        # Missing pattern
        missing_by_target = self.data.groupby(self.data[target_column].isnull()).size()
        fig.add_trace(
            go.Pie(
                labels=['Not Missing', 'Missing'],
                values=missing_by_target.values,
                name='Missing Pattern'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text=f"Target Variable Analysis: {target_column}",
            template=self.theme
        )
        
        return fig
    
    def _create_relationship_plots(self, numeric_cols: List[str]) -> go.Figure:
        """Create relationship plots between numeric variables."""
        if len(numeric_cols) < 2:
            return go.Figure()
        
        # Create scatter plot matrix for first few columns
        cols_to_plot = numeric_cols[:4]  # Limit to 4 columns
        
        fig = make_subplots(
            rows=len(cols_to_plot), cols=len(cols_to_plot),
            subplot_titles=cols_to_plot,
            shared_xaxes=True,
            shared_yaxes=True
        )
        
        for i, col1 in enumerate(cols_to_plot):
            for j, col2 in enumerate(cols_to_plot):
                if i == j:
                    # Histogram on diagonal
                    fig.add_trace(
                        go.Histogram(
                            x=self.data[col1].dropna(),
                            name=col1,
                            opacity=0.7
                        ),
                        row=i+1, col=j+1
                    )
                else:
                    # Scatter plot off-diagonal
                    fig.add_trace(
                        go.Scatter(
                            x=self.data[col2].dropna(),
                            y=self.data[col1].dropna(),
                            mode='markers',
                            name=f'{col1} vs {col2}',
                            opacity=0.6
                        ),
                        row=i+1, col=j+1
                    )
        
        fig.update_layout(
            height=300 * len(cols_to_plot),
            title_text="Relationship Matrix",
            template=self.theme
        )
        
        return fig
    
    def _create_time_series_plots(self, datetime_cols: List[str]) -> go.Figure:
        """Create time series plots."""
        fig = make_subplots(
            rows=len(datetime_cols), cols=1,
            subplot_titles=datetime_cols,
            shared_xaxes=True
        )
        
        for i, col in enumerate(datetime_cols):
            # Sort by datetime
            temp_data = self.data.sort_values(col)
            
            # Plot count over time
            time_counts = temp_data.groupby(temp_data[col].dt.date).size()
            
            fig.add_trace(
                go.Scatter(
                    x=time_counts.index,
                    y=time_counts.values,
                    mode='lines+markers',
                    name=f'Records over {col}'
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            height=300 * len(datetime_cols),
            title_text="Time Series Analysis",
            template=self.theme
        )
        
        return fig
    
    @handle_errors("data")
    def create_ml_visualizations(self, model_results: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create ML model visualization."""
        visualizations = {}
        
        # Model comparison
        if 'model_comparison' in model_results:
            visualizations['model_comparison'] = self._create_model_comparison_chart(model_results['model_comparison'])
        
        # Feature importance
        if 'feature_importance' in model_results:
            visualizations['feature_importance'] = self._create_feature_importance_chart(model_results['feature_importance'])
        
        # Learning curves
        if 'learning_curves' in model_results:
            visualizations['learning_curves'] = self._create_learning_curves_chart(model_results['learning_curves'])
        
        # Confusion matrix
        if 'confusion_matrix' in model_results:
            visualizations['confusion_matrix'] = self._create_confusion_matrix_chart(model_results['confusion_matrix'])
        
        # ROC curves
        if 'roc_curves' in model_results:
            visualizations['roc_curves'] = self._create_roc_curves_chart(model_results['roc_curves'])
        
        return visualizations
    
    def _create_model_comparison_chart(self, comparison_data: pd.DataFrame) -> go.Figure:
        """Create model comparison chart."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Performance Metrics', 'Training Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Performance metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for metric in metrics:
            if metric in comparison_data.columns:
                fig.add_trace(
                    go.Bar(
                        x=comparison_data['Model'],
                        y=comparison_data[metric],
                        name=metric,
                        opacity=0.8
                    ),
                    row=1, col=1
                )
        
        # Training time
        if 'Training_Time' in comparison_data.columns:
            fig.add_trace(
                go.Bar(
                    x=comparison_data['Model'],
                    y=comparison_data['Training_Time'],
                    name='Training Time',
                    marker_color='orange',
                    opacity=0.8
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=500,
            title_text="Model Comparison",
            template=self.theme,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def _create_feature_importance_chart(self, importance_data: Dict[str, float]) -> go.Figure:
        """Create feature importance chart."""
        # Sort features by importance
        sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_features[:20])  # Top 20 features
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(importance),
                y=list(features),
                orientation='h',
                marker_color=self.color_palette[0],
                text=list(importance),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(features) * 25),
            template=self.theme
        )
        
        return fig
    
    def _create_learning_curves_chart(self, learning_data: Dict[str, Any]) -> go.Figure:
        """Create learning curves chart."""
        fig = go.Figure()
        
        # Training scores
        if 'train_scores' in learning_data:
            fig.add_trace(go.Scatter(
                x=learning_data.get('train_sizes', []),
                y=learning_data['train_scores'],
                mode='lines+markers',
                name='Training Score',
                line=dict(color='blue')
            ))
        
        # Validation scores
        if 'val_scores' in learning_data:
            fig.add_trace(go.Scatter(
                x=learning_data.get('train_sizes', []),
                y=learning_data['val_scores'],
                mode='lines+markers',
                name='Validation Score',
                line=dict(color='red')
            ))
        
        fig.update_layout(
            title="Learning Curves",
            xaxis_title="Training Set Size",
            yaxis_title="Score",
            template=self.theme
        )
        
        return fig
    
    def _create_confusion_matrix_chart(self, confusion_data: np.ndarray) -> go.Figure:
        """Create confusion matrix heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=confusion_data,
            colorscale='Blues',
            showscale=True,
            text=confusion_data,
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            template=self.theme
        )
        
        return fig
    
    def _create_roc_curves_chart(self, roc_data: Dict[str, Any]) -> go.Figure:
        """Create ROC curves chart."""
        fig = go.Figure()
        
        for model_name, roc_info in roc_data.items():
            if 'fpr' in roc_info and 'tpr' in roc_info:
                fig.add_trace(go.Scatter(
                    x=roc_info['fpr'],
                    y=roc_info['tpr'],
                    mode='lines',
                    name=f'{model_name} (AUC = {roc_info.get("auc", 0):.3f})',
                    line=dict(width=2)
                ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title="ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template=self.theme
        )
        
        return fig
    
    @handle_errors("data")
    def create_text_visualizations(self, text_data: pd.Series) -> Dict[str, go.Figure]:
        """Create text analytics visualizations."""
        visualizations = {}
        
        # Word frequency
        visualizations['word_frequency'] = self._create_word_frequency_chart(text_data)
        
        # N-gram analysis
        visualizations['ngram_analysis'] = self._create_ngram_chart(text_data)
        
        # Text length distribution
        visualizations['text_length'] = self._create_text_length_distribution(text_data)
        
        # Sentiment analysis (if applicable)
        visualizations['sentiment'] = self._create_sentiment_analysis(text_data)
        
        return visualizations
    
    def _create_word_frequency_chart(self, text_data: pd.Series) -> go.Figure:
        """Create word frequency chart."""
        from collections import Counter
        
        # Combine all text and count words
        all_text = ' '.join(text_data.dropna().astype(str))
        words = all_text.lower().split()
        word_freq = Counter(words)
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered_freq = {word: count for word, count in word_freq.items() if word not in stop_words and len(word) > 2}
        
        # Get top 20 words
        top_words = sorted(filtered_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        words, counts = zip(*top_words)
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(counts),
                y=list(words),
                orientation='h',
                marker_color=self.color_palette[1],
                text=list(counts),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Top 20 Word Frequencies",
            xaxis_title="Frequency",
            yaxis_title="Words",
            height=max(400, len(words) * 25),
            template=self.theme
        )
        
        return fig
    
    def _create_ngram_chart(self, text_data: pd.Series, n: int = 2) -> go.Figure:
        """Create n-gram analysis chart."""
        from collections import Counter
        
        # Generate n-grams
        all_text = ' '.join(text_data.dropna().astype(str)).lower()
        words = all_text.split()
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
        
        ngram_freq = Counter(ngrams)
        top_ngrams = sorted(ngram_freq.items(), key=lambda x: x[1], reverse=True)[:15]
        ngrams, counts = zip(*top_ngrams)
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(counts),
                y=list(ngrams),
                orientation='h',
                marker_color=self.color_palette[2],
                text=list(counts),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f"Top 15 {n}-gram Frequencies",
            xaxis_title="Frequency",
            yaxis_title=f"{n}-grams",
            height=max(400, len(ngrams) * 25),
            template=self.theme
        )
        
        return fig
    
    def _create_text_length_distribution(self, text_data: pd.Series) -> go.Figure:
        """Create text length distribution."""
        text_lengths = text_data.dropna().astype(str).str.len()
        
        fig = go.Figure(data=[
            go.Histogram(
                x=text_lengths,
                nbinsx=30,
                marker_color=self.color_palette[3],
                opacity=0.7
            )
        ])
        
        fig.update_layout(
            title="Text Length Distribution",
            xaxis_title="Text Length (characters)",
            yaxis_title="Frequency",
            template=self.theme
        )
        
        return fig
    
    def _create_sentiment_analysis(self, text_data: pd.Series) -> go.Figure:
        """Create sentiment analysis visualization."""
        try:
            from textblob import TextBlob
            
            # Calculate sentiment scores
            sentiments = []
            for text in text_data.dropna().astype(str):
                blob = TextBlob(text)
                sentiments.append(blob.sentiment.polarity)
            
            # Create sentiment categories
            sentiment_categories = []
            for sentiment in sentiments:
                if sentiment > 0.1:
                    sentiment_categories.append('Positive')
                elif sentiment < -0.1:
                    sentiment_categories.append('Negative')
                else:
                    sentiment_categories.append('Neutral')
            
            # Count categories
            from collections import Counter
            sentiment_counts = Counter(sentiment_categories)
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(sentiment_counts.keys()),
                    values=list(sentiment_counts.values()),
                    marker_colors=['green', 'gray', 'red']
                )
            ])
            
            fig.update_layout(
                title="Sentiment Analysis Distribution",
                template=self.theme
            )
            
            return fig
            
        except ImportError:
            # TextBlob not available
            fig = go.Figure()
            fig.add_annotation(
                text="TextBlob not available for sentiment analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle'
            )
            return fig
    
    @handle_errors("data")
    def create_clustering_visualizations(self, cluster_data: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create clustering visualization."""
        visualizations = {}
        
        # Cluster scatter plot
        if 'cluster_labels' in cluster_data and 'features' in cluster_data:
            visualizations['cluster_scatter'] = self._create_cluster_scatter_plot(cluster_data)
        
        # Cluster centers
        if 'cluster_centers' in cluster_data:
            visualizations['cluster_centers'] = self._create_cluster_centers_plot(cluster_data)
        
        # Silhouette analysis
        if 'silhouette_scores' in cluster_data:
            visualizations['silhouette'] = self._create_silhouette_plot(cluster_data)
        
        return visualizations
    
    def _create_cluster_scatter_plot(self, cluster_data: Dict[str, Any]) -> go.Figure:
        """Create cluster scatter plot."""
        features = cluster_data['features']
        labels = cluster_data['cluster_labels']
        
        # Use PCA for visualization if more than 2 dimensions
        if features.shape[1] > 2:
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
        else:
            features_2d = features
        
        fig = go.Figure()
        
        # Plot each cluster
        unique_labels = np.unique(labels)
        colors = self.color_palette[:len(unique_labels)]
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            fig.add_trace(go.Scatter(
                x=features_2d[mask, 0],
                y=features_2d[mask, 1],
                mode='markers',
                name=f'Cluster {label}',
                marker=dict(color=colors[i], size=8, opacity=0.7)
            ))
        
        fig.update_layout(
            title="Cluster Visualization",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            template=self.theme
        )
        
        return fig
    
    def _create_cluster_centers_plot(self, cluster_data: Dict[str, Any]) -> go.Figure:
        """Create cluster centers visualization."""
        centers = cluster_data['cluster_centers']
        
        fig = go.Figure(data=go.Heatmap(
            z=centers,
            colorscale='Viridis',
            showscale=True
        ))
        
        fig.update_layout(
            title="Cluster Centers",
            xaxis_title="Features",
            yaxis_title="Clusters",
            template=self.theme
        )
        
        return fig
    
    def _create_silhouette_plot(self, cluster_data: Dict[str, Any]) -> go.Figure:
        """Create silhouette analysis plot."""
        silhouette_scores = cluster_data['silhouette_scores']
        labels = cluster_data['cluster_labels']
        
        fig = go.Figure()
        
        # Create silhouette plot for each cluster
        unique_labels = np.unique(labels)
        y_lower = 10
        
        for i, label in enumerate(unique_labels):
            cluster_silhouette = silhouette_scores[labels == label]
            cluster_silhouette.sort()
            
            size_cluster_i = cluster_silhouette.shape[0]
            y_upper = y_lower + size_cluster_i
            
            fig.add_trace(go.Scatter(
                x=cluster_silhouette,
                y=list(range(y_lower, y_upper)),
                fill='toself',
                name=f'Cluster {label}',
                mode='lines'
            ))
            
            y_lower = y_upper + 10  # 10 for space between clusters
        
        # Add average silhouette line
        avg_silhouette = np.mean(silhouette_scores)
        fig.add_vline(
            x=avg_silhouette,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average: {avg_silhouette:.3f}"
        )
        
        fig.update_layout(
            title="Silhouette Analysis",
            xaxis_title="Silhouette Coefficient",
            yaxis_title="Cluster",
            template=self.theme
        )
        
        return fig
    
    @rate_limit()
    def export_visualization(self, fig: go.Figure, filename: str, format: str = 'html') -> str:
        """Export visualization to file."""
        if format == 'html':
            filepath = f"exports/{filename}.html"
            fig.write_html(filepath)
        elif format == 'png':
            filepath = f"exports/{filename}.png"
            fig.write_image(filepath)
        elif format == 'pdf':
            filepath = f"exports/{filename}.pdf"
            fig.write_image(filepath)
        else:
            raise ValidationError(f"Unsupported export format: {format}")
        
        return filepath
    
    def create_interactive_dashboard(self, dashboard_config: Dict[str, Any]) -> go.Figure:
        """Create interactive dashboard with multiple visualizations."""
        # This would create a comprehensive dashboard
        # Implementation depends on specific requirements
        pass


class VisualizationGenerator:
    """Automated visualization generator based on data characteristics."""
    
    def __init__(self):
        """Initialize visualization generator."""
        self.visualizer = AdvancedVisualizer()
        self.type_detector = DataTypeDetector()
    
    def auto_generate_visualizations(self, data: pd.DataFrame, analysis_type: str = 'auto') -> Dict[str, go.Figure]:
        """Automatically generate appropriate visualizations."""
        self.visualizer.set_data(data)
        
        if analysis_type == 'auto':
            analysis_type = self._determine_analysis_type(data)
        
        visualizations = {}
        
        if analysis_type == 'eda':
            visualizations = self.visualizer.create_eda_visualizations()
        elif analysis_type == 'ml':
            # Would need model results
            pass
        elif analysis_type == 'text':
            # Would need text data
            pass
        elif analysis_type == 'clustering':
            # Would need clustering results
            pass
        
        return visualizations
    
    def _determine_analysis_type(self, data: pd.DataFrame) -> str:
        """Determine appropriate analysis type based on data."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        text_cols = [col for col in categorical_cols if data[col].dtype == 'object']
        
        # Simple heuristic for analysis type
        if len(text_cols) > 0 and data[text_cols[0]].str.len().mean() > 50:
            return 'text'
        elif len(numeric_cols) > 2:
            return 'eda'
        else:
            return 'eda'


# Standalone functions for backward compatibility with app_adapted.py
def create_scatter_plot(data, x_col, y_col, color_by=None):
    """Create scatter plot."""
    try:
        import plotly.express as px
        
        if color_by and color_by in data.columns:
            fig = px.scatter(data, x=x_col, y=y_col, color=color_by,
                           title=f"{y_col} vs {x_col}")
        else:
            fig = px.scatter(data, x=x_col, y=y_col,
                           title=f"{y_col} vs {x_col}")
        
        fig.update_layout(template='plotly_white')
        return fig
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
        return None


def create_bar_chart(data, x_col, y_col):
    """Create bar chart."""
    try:
        import plotly.express as px
        fig = px.bar(data, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        fig.update_layout(template='plotly_white')
        return fig
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        return None


def create_histogram(data, col):
    """Create histogram."""
    try:
        import plotly.express as px
        fig = px.histogram(data, x=col, title=f"Distribution of {col}")
        fig.update_layout(template='plotly_white')
        return fig
    except Exception as e:
        print(f"Error creating histogram: {e}")
        return None


def create_box_plot(data, col):
    """Create box plot."""
    try:
        import plotly.express as px
        fig = px.box(data, y=col, title=f"Box Plot of {col}")
        fig.update_layout(template='plotly_white')
        return fig
    except Exception as e:
        print(f"Error creating box plot: {e}")
        return None


def create_correlation_heatmap(data):
    """Create correlation heatmap."""
    try:
        import plotly.express as px
        import numpy as np
        
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return None
            
        corr_matrix = numeric_data.corr()
        
        fig = px.imshow(corr_matrix,
                       title="Correlation Matrix",
                       color_continuous_scale='RdBu',
                       aspect="auto")
        fig.update_layout(template='plotly_white')
        return fig
    except Exception as e:
        print(f"Error creating correlation heatmap: {e}")
        return None


def create_line_chart(data, x_col, y_col):
    """Create line chart."""
    try:
        import plotly.express as px
        fig = px.line(data, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
        fig.update_layout(template='plotly_white')
        return fig
    except Exception as e:
        print(f"Error creating line chart: {e}")
        return None


def create_violin_plot(data, num_col, cat_col):
    """Create violin plot."""
    try:
        import plotly.express as px
        fig = px.violin(data, x=cat_col, y=num_col,
                       title=f"Distribution of {num_col} by {cat_col}")
        fig.update_layout(template='plotly_white')
        return fig
    except Exception as e:
        print(f"Error creating violin plot: {e}")
        return None


def create_confusion_matrix_plot(confusion_matrix):
    """Create confusion matrix plot."""
    try:
        import plotly.express as px
        fig = px.imshow(confusion_matrix,
                       title="Confusion Matrix",
                       color_continuous_scale='Blues',
                       labels=dict(x="Predicted", y="Actual", color="Count"))
        fig.update_layout(template='plotly_white')
        return fig
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        return None


def create_feature_importance_plot(importance_data):
    """Create feature importance plot."""
    try:
        import plotly.express as px
        import pandas as pd
        
        if isinstance(importance_data, dict):
            df = pd.DataFrame(list(importance_data.items()),
                             columns=['Feature', 'Importance'])
        else:
            df = importance_data
            
        df = df.sort_values('Importance', ascending=True).tail(20)
        
        fig = px.bar(df, x='Importance', y='Feature',
                    title="Feature Importance", orientation='h')
        fig.update_layout(template='plotly_white')
        return fig
    except Exception as e:
        print(f"Error creating feature importance plot: {e}")
        return None