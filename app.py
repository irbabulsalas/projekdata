import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

from modules.data_processing import load_data_from_file, profile_data, clean_data
from modules.ml_models import train_classification_models, train_regression_models, perform_clustering, perform_pca, get_feature_importance
from modules.text_analytics import analyze_text_column, analyze_sentiment
from modules.visualizations import (
    create_scatter_plot, create_bar_chart, create_histogram, create_box_plot,
    create_correlation_heatmap, create_line_chart, create_violin_plot,
    create_confusion_matrix_plot, create_feature_importance_plot
)
from modules.gemini_integration import chat_with_gemini, get_data_context
from modules.export_handler import create_export_center
from modules.deep_learning import deep_learning, DeepLearningModels
from modules.advanced_timeseries import advanced_timeseries, AdvancedTimeSeriesModels
from database.auth_manager import auth_manager, AuthManager
from database.session_manager import session_manager, SessionManager
from utils.helpers import initialize_session_state, get_numeric_columns, get_categorical_columns, calculate_data_quality_score, validate_dataframe
from utils.rate_limiter import initialize_rate_limiter, can_make_request, show_rate_limit_status, update_rate_limit, get_remaining_requests
from utils.error_handler import safe_execute

# Production optimizations for Railway deployment
import sys
import os

# Set environment variables for production
if os.getenv('RAILWAY_ENVIRONMENT') == 'production':
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'

st.set_page_config(
    page_title="AI Data Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            display: none;
        }
        section[data-testid="stSidebar"][aria-expanded="true"] {
            display: block;
            width: 100% !important;
        }
        .main .block-container {
            padding: 1rem !important;
            max-width: 100% !important;
        }
        div[data-testid="column"] {
            width: 100% !important;
            margin-bottom: 1rem;
        }
    }
    
    @media (min-width: 769px) and (max-width: 1024px) {
        section[data-testid="stSidebar"] {
            width: 250px !important;
        }
    }
    
    @media (min-width: 1025px) {
        .main .block-container {
            max-width: 1200px;
            padding: 2rem 3rem;
        }
    }
    
    .stButton button {
        width: 100%;
    }
    
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    h1, h2, h3 {
        color: #4A90E2;
    }
    
    .profile-header {
        display: flex;
        align-items: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

initialize_session_state()
initialize_rate_limiter()

try:
    from database.db_manager import db_manager
    if db_manager is not None:
        db_manager.create_tables()
        AuthManager.init_session()
except Exception as e:
    st.sidebar.warning(f"⚠️ Database features unavailable: {str(e)}")
    pass

def show_header():
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col1:
        if os.path.exists("assets/profile_photo.jpg"):
            st.image("assets/profile_photo.jpg", width=80)
    
    with col2:
        st.markdown("""
        <div style='padding-top: 10px;'>
            <h1 style='margin:0; color: #4A90E2;'>Muhammad Irbabul Salas</h1>
            <p style='margin:0; color: #7F8C8D; font-size: 14px;'>
                AI-Powered Data Analysis Platform | Automated ML & Insights with Gemini 2.5
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.button("🌙" if st.session_state.get('theme') == 'light' else "☀️"):
            st.session_state.theme = 'dark' if st.session_state.get('theme') == 'light' else 'light'
            st.rerun()

show_header()

with st.sidebar:
    st.title("📊 Navigation")
    
    pages = ["📈 Overview", "🔍 Data Profiling", "📊 EDA", "🤖 ML Models", "🚀 Advanced ML", "⏰ Time Series", "🧠 Deep Learning", "📈 Advanced TS", "📝 Text Analytics", "💾 Projects", "📥 Export Center"]
    st.session_state.current_page = st.radio("Go to", pages, label_visibility="collapsed")
    
    st.markdown("---")
    st.subheader("📤 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=['csv', 'xlsx', 'json', 'parquet', 'txt'],
        help="Supported: CSV, Excel, JSON, Parquet, TSV"
    )
    
    if uploaded_file:
        df = load_data_from_file(uploaded_file)
        if df is not None:
            st.session_state.uploaded_data = df.copy()  # Store original data
            st.session_state.cleaned_data = df.copy()  # Store working data
    
    if st.button("📝 Load Sample E-commerce Data"):
        try:
            df = pd.read_csv("assets/sample_datasets/sample_ecommerce.csv")
            st.session_state.uploaded_data = df.copy()  # Store original data
            st.session_state.cleaned_data = df.copy()  # Store working data
            st.success("✅ Sample data loaded!")
            st.rerun()
        except:
            st.error("❌ Sample data not found")
    
    if st.button("💬 Load Sample Reviews Data"):
        try:
            df = pd.read_csv("assets/sample_datasets/sample_reviews.csv")
            st.session_state.uploaded_data = df.copy()  # Store original data
            st.session_state.cleaned_data = df.copy()  # Store working data
            st.success("✅ Sample reviews loaded!")
            st.rerun()
        except:
            st.error("❌ Sample data not found")
    
    st.markdown("---")
    st.subheader("❓ Help & Support")
    
    with st.expander("📖 Quick Guide"):
        st.write("""
        **Getting Started:**
        1. Upload your data or try sample datasets
        2. Explore different dashboards
        3. Ask AI for insights via chat
        4. Export your results
        
        **Tips:**
        - Use AI chat for complex analysis
        - Download charts as PNG/HTML
        - Save trained models for deployment
        """)
    
    with st.expander("⌨️ Keyboard Shortcuts"):
        st.write("""
        - `Ctrl+U`: Upload data
        - `Ctrl+/`: Search help
        - `Ctrl+Enter`: Send chat message
        """)
    
    AuthManager.render_auth_sidebar()

df = st.session_state.get('cleaned_data')

if st.session_state.current_page == "📈 Overview":
    st.title("📈 Overview Dashboard")
    
    if df is not None and not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📁 Total Rows", f"{len(df):,}")
        with col2:
            st.metric("📊 Columns", len(df.columns))
        with col3:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.metric("⚠️ Missing", f"{missing_pct:.1f}%")
        with col4:
            quality_score = calculate_data_quality_score(df)
            st.metric("✅ Quality Score", f"{quality_score}%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Column Types")
            type_counts = {
                'Numeric': len(get_numeric_columns(df)),
                'Categorical': len(get_categorical_columns(df)),
                'Other': len(df.columns) - len(get_numeric_columns(df)) - len(get_categorical_columns(df))
            }
            
            type_df = pd.DataFrame(list(type_counts.items()), columns=['Type', 'Count'])
            fig = create_bar_chart(type_df, 'Type', 'Count')
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="overview_column_types")
        
        with col2:
            st.subheader("📈 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Showing 10 of {len(df):,} rows")
        
        st.markdown("---")
        st.subheader("🤖 AI Quick Insights")
        
        if st.button("✨ Generate Insights with AI"):
            with st.spinner("Analyzing your data..."):
                try:
                    from modules.gemini_integration import generate_insights
                    insights = safe_execute("AI Insights", generate_insights, df, "general")
                except ImportError:
                    st.error("AI Insights module not available. Please check your installation.")
                    insights = None
                if insights:
                    st.success(insights)
    else:
        st.info("📤 Please upload a dataset or load sample data from the sidebar to begin analysis.")

elif st.session_state.current_page == "🔍 Data Profiling":
    st.title("🔍 Data Profiling")
    
    # CRITICAL FIX: Always get fresh data from session state
    df = st.session_state.get('cleaned_data')
    
    
    if validate_dataframe(df):
        # Check if we have original data for comparison
        original_df = st.session_state.get('uploaded_data')
        
        # More robust check for cleaned data
        is_cleaned = False
        if original_df is not None:
            # Check if shapes are different
            if original_df.shape != df.shape:
                is_cleaned = True
            # Check if data content is different
            elif not original_df.equals(df):
                is_cleaned = True
            else:
                pass
        
        # Additional check: if cleaning_success flag is set, force is_cleaned to True
        if st.session_state.get('cleaning_success', False):
            is_cleaned = True
        
        # Show persistent success message if cleaning was just performed
        if st.session_state.get('cleaning_success', False):
            st.success("✅ Data cleaned successfully!")
            st.write("**Cleaning Summary:**")
            for action in st.session_state.get('cleaning_report', []):
                st.write(f"• {action}")
            st.write(f"**Data Shape:** {st.session_state.get('original_shape', 'N/A')} → {st.session_state.get('cleaned_shape', 'N/A')}")
            
            # Force is_cleaned to True when cleaning_success is True
            is_cleaned = True
            
            # Clear the flag after displaying
            st.session_state.cleaning_success = False
        
        # Debug information
        if is_cleaned:
            st.info("✨ Data comparison mode: Showing before/after cleaning views")
        else:
            st.info("📊 Showing current data only. Clean data to see comparison views.")
        
        # Create tabs for before/after comparison
        if is_cleaned:
            tab1, tab2, tab3 = st.tabs(["📊 Current Data", "🔍 Before Cleaning", "📈 Comparison"])
        else:
            tab1, tab2 = st.tabs(["📊 Current Data", "🧹 Data Cleaning"])
        
        with tab1:
            # Profile current (cleaned) data - ensure fresh data
            current_df = st.session_state.get('cleaned_data')
            current_profile = safe_execute("Current Data Profiling", profile_data, current_df)
            
            if current_profile:
                st.subheader("📋 Basic Information (Current Data)")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Rows", f"{current_profile['basic_info']['rows']:,}")
                with col2:
                    st.metric("Columns", current_profile['basic_info']['columns'])
                with col3:
                    st.metric("Duplicates", current_profile['basic_info']['duplicates'])
                with col4:
                    st.metric("Quality", f"{current_profile['quality_score']}%")
                
                st.markdown("---")
                
                st.subheader("📊 Column Details (Current Data)")
                current_col_info_df = pd.DataFrame(current_profile['column_info']).T
                st.dataframe(current_col_info_df, use_container_width=True)
                
                st.markdown("---")
                
                st.subheader("📈 Correlation Heatmap (Current Data)")
                fig = create_correlation_heatmap(current_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="correlation_current")
        
        # Show before cleaning data if available
        if is_cleaned:
            with tab2:
                st.subheader("🔍 Data Before Cleaning")
                # Ensure fresh original data
                original_df_fresh = st.session_state.get('uploaded_data')
                original_profile = safe_execute("Original Data Profiling", profile_data, original_df_fresh)
                
                if original_profile:
                    st.subheader("📋 Basic Information (Before Cleaning)")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Rows", f"{original_profile['basic_info']['rows']:,}")
                    with col2:
                        st.metric("Columns", original_profile['basic_info']['columns'])
                    with col3:
                        st.metric("Duplicates", original_profile['basic_info']['duplicates'])
                    with col4:
                        st.metric("Quality", f"{original_profile['quality_score']}%")
                    
                    st.markdown("---")
                    
                    st.subheader("📊 Column Details (Before Cleaning)")
                    original_col_info_df = pd.DataFrame(original_profile['column_info']).T
                    st.dataframe(original_col_info_df, use_container_width=True)
                    
                    st.markdown("---")
                    
                    st.subheader("📈 Correlation Heatmap (Before Cleaning)")
                    original_fig = create_correlation_heatmap(original_df_fresh)
                    if original_fig:
                        st.plotly_chart(original_fig, use_container_width=True, key="correlation_before")
            
            with tab3:
                st.subheader("📈 Before vs After Comparison")
                
                # CRITICAL: Ensure we have the correct data
                # Get fresh data from session state
                current_df = st.session_state.get('cleaned_data')
                original_df = st.session_state.get('uploaded_data')
                
                
                # Use the fresh data for calculations
                rows_before = len(original_df) if original_df is not None else 0
                rows_after = len(current_df) if current_df is not None else 0
                cols_before = len(original_df.columns) if original_df is not None else 0
                cols_after = len(current_df.columns) if current_df is not None else 0
                duplicates_before = original_df.duplicated().sum() if original_df is not None else 0
                duplicates_after = current_df.duplicated().sum() if current_df is not None else 0
                
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Rows After Cleaning",
                        f"{rows_after:,}",
                        delta=f"{rows_after - rows_before:,}"
                    )
                
                with col2:
                    st.metric(
                        "Columns After Cleaning",
                        cols_after,
                        delta=cols_after - cols_before
                    )
                
                with col3:
                    duplicates_removed = duplicates_before - duplicates_after
                    st.metric(
                        "Duplicates Removed",
                        duplicates_removed,
                        delta=f"-{duplicates_removed}"
                    )
                
                st.markdown("---")
                
                # Missing values comparison
                st.subheader("🔍 Missing Values Comparison")
                missing_before = original_df.isnull().sum()
                missing_after = df.isnull().sum()
                
                missing_comparison = pd.DataFrame({
                    'Before': missing_before,
                    'After': missing_after,
                    'Change': missing_after - missing_before
                })
                
                # Only show columns with missing values
                missing_cols = missing_comparison[(missing_before > 0) | (missing_after > 0)]
                
                if not missing_cols.empty:
                    st.dataframe(missing_cols, use_container_width=True)
                else:
                    st.success("✅ No missing values in either dataset!")
                
                st.markdown("---")
                
                # Data types comparison
                st.subheader("📊 Data Types Comparison")
                dtypes_before = original_df.dtypes.value_counts()
                dtypes_after = df.dtypes.value_counts()
                
                dtype_comparison = pd.DataFrame({
                    'Before': dtypes_before,
                    'After': dtypes_after
                }).fillna(0).astype(int)
                
                st.dataframe(dtype_comparison, use_container_width=True)
        
        # Data cleaning controls (in both scenarios)
        if is_cleaned:
            with tab2:
                st.markdown("---")
                st.subheader("🧹 Data Cleaning Options")
                
                # Add reset and refresh buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Reset to Original Data", type="secondary"):
                        st.session_state.cleaned_data = st.session_state.uploaded_data.copy()
                        st.success("✅ Data reset to original!")
                        st.rerun()
                with col2:
                    if st.button("🔄 Refresh Page", type="secondary"):
                        st.rerun()
        else:
            with tab2:
                st.subheader("🧹 Data Cleaning")
        
        # Common cleaning controls
        handle_missing = st.selectbox(
            "Handle Missing Values",
            ["mean", "median", "mode", "drop"],
            help="Strategy for missing values"
        )
        
        remove_dup = st.checkbox("Remove Duplicates", value=True)
        
        handle_outliers = st.selectbox(
            "Handle Outliers",
            ["none", "iqr", "zscore"],
            help="Outlier detection method"
        )
        
        # Add refresh button and clean button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data", type="secondary"):
                st.rerun()
        
        with col2:
            if st.button("🧹 Clean Data Now", type="primary"):
                # Use original data if available, otherwise use current data
                data_to_clean = original_df if original_df is not None else df
                
                # Debug info
                
                cleaned_df, report = safe_execute(
                    "Data Cleaning",
                    clean_data,
                    data_to_clean,
                    handle_missing,
                    remove_dup,
                    handle_outliers
                )
                
                if cleaned_df is not None:
                    # Always save original data if not already saved
                    if st.session_state.get('uploaded_data') is None:
                        st.session_state.uploaded_data = data_to_clean.copy()
                    
                    # Save cleaned data with debug info - FORCE UPDATE
                    st.session_state['cleaned_data'] = cleaned_df.copy()  # Use .copy() and explicit key
                    
                    # Create a persistent success message
                    st.session_state.cleaning_success = True
                    st.session_state.cleaning_report = report.get('actions', [])
                    st.session_state.original_shape = st.session_state.uploaded_data.shape
                    st.session_state.cleaned_shape = cleaned_df.shape
                    
                    # Debug: Verify session state before rerun
                    
                    # Force rerun to refresh the page with new tabs
                    st.rerun()
                else:
                    st.error("❌ Cleaning failed! Please check your data and try again.")

elif st.session_state.current_page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    
    # FIX: Always get fresh data from session state
    df = st.session_state.get('cleaned_data')
    
    if validate_dataframe(df):
        tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🎯 Relationships", "📦 Comparisons"])
        
        with tab1:
            numeric_cols = get_numeric_columns(df)
            
            if numeric_cols:
                selected_col = st.selectbox("Select Column", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Histogram")
                    fig = create_histogram(df, selected_col)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="eda_histogram")
                
                with col2:
                    st.subheader("📦 Box Plot")
                    fig = create_box_plot(df, selected_col)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="eda_box_plot")
            else:
                st.warning("No numeric columns available")
        
        with tab2:
            numeric_cols = get_numeric_columns(df)
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
                with col2:
                    y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col], key="scatter_y")
                
                categorical_cols = get_categorical_columns(df)
                color_by = st.selectbox("Color by (optional)", [None] + categorical_cols)
                
                fig = create_scatter_plot(df, x_col, y_col, color_by)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="eda_scatter_plot")
            else:
                st.warning("Need at least 2 numeric columns")
        
        with tab3:
            categorical_cols = get_categorical_columns(df)
            numeric_cols = get_numeric_columns(df)
            
            if categorical_cols and numeric_cols:
                cat_col = st.selectbox("Categorical Column", categorical_cols)
                num_col = st.selectbox("Numeric Column", numeric_cols)
                
                st.subheader("🎻 Violin Plot")
                fig = create_violin_plot(df, num_col, cat_col)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="eda_violin_plot")
            else:
                st.warning("Need both categorical and numeric columns")

elif st.session_state.current_page == "🤖 ML Models":
    st.title("🤖 Machine Learning Models")
    
    # FIX: Always get fresh data from session state
    df = st.session_state.get('cleaned_data')
    
    if validate_dataframe(df, min_rows=50):
        tab1, tab2, tab3 = st.tabs(["🎯 Classification", "📈 Regression", "🔍 Clustering"])
        
        with tab1:
            st.subheader("Classification Models")
            
            all_cols = df.columns.tolist()
            target_col = st.selectbox("Select Target Column", all_cols)
            
            model_options = st.multiselect(
                "Select Models",
                ["random_forest", "xgboost", "logistic", "lightgbm"],
                default=["random_forest", "xgboost"]
            )
            
            if st.button("🚀 Train Classification Models"):
                with st.spinner("Training models..."):
                    results = safe_execute(
                        "Classification",
                        train_classification_models,
                        df,
                        target_col,
                        model_options
                    )
                    
                    if results:
                        st.session_state.trained_models = results
                        
                        st.success(f"✅ Trained {len(results)} models!")
                        
                        metrics_data = []
                        for name, data in results.items():
                            row = {'Model': name.upper()}
                            # Check if data is a dictionary and has 'metrics' key
                            if isinstance(data, dict) and 'metrics' in data:
                                row.update(data['metrics'])
                            elif isinstance(data, dict):
                                # If data is dict but no 'metrics' key, update with all data
                                row.update(data)
                            metrics_data.append(row)
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df.round(4), use_container_width=True)
                        
                        # Safe best model selection with error handling
                        importance = None  # Initialize importance variable
                        try:
                            best_model = max(results.items(), key=lambda x: x[1]['metrics']['accuracy'])
                            accuracy = best_model[1]['metrics']['accuracy']
                            st.info(f"⭐ Best Model: **{best_model[0].upper()}** with {accuracy:.2%} accuracy")
                            
                            if 'confusion_matrix' in best_model[1]:
                                st.subheader("📊 Confusion Matrix (Best Model)")
                                cm_fig = create_confusion_matrix_plot(np.array(best_model[1]['confusion_matrix']))
                                if cm_fig:
                                    st.plotly_chart(cm_fig, use_container_width=True, key="ml_confusion_matrix")
                            
                            numeric_cols = get_numeric_columns(df.drop(columns=[target_col]))
                            importance = get_feature_importance(best_model[1]['model'], numeric_cols)
                        except (KeyError, TypeError) as e:
                            st.warning(f"Could not determine best model: {str(e)}")
                            # Use first model as fallback
                            if results:
                                best_model = list(results.items())[0]
                                st.info(f"⭐ Using Model: **{best_model[0].upper()}**")
                                # Try to get importance for fallback model
                                try:
                                    numeric_cols = get_numeric_columns(df.drop(columns=[target_col]))
                                    importance = get_feature_importance(best_model[1]['model'], numeric_cols)
                                except:
                                    pass
                        
                        if importance:
                            st.subheader("🎯 Feature Importance")
                            fig = create_feature_importance_plot(importance)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key="ml_feature_importance")
        with tab2:
            st.subheader("Regression Models")
            
            numeric_cols = get_numeric_columns(df)
            if numeric_cols:
                target_col = st.selectbox("Select Target Column", numeric_cols, key="reg_target")
                
                model_options = st.multiselect(
                    "Select Models",
                    ["random_forest", "xgboost", "ridge", "lasso"],
                    default=["random_forest", "ridge"],
                    key="reg_models"
                )
                
                if st.button("🚀 Train Regression Models"):
                    with st.spinner("Training models..."):
                        results = safe_execute(
                            "Regression",
                            train_regression_models,
                            df,
                            target_col,
                            model_options
                        )
                        
                        if results:
                            metrics_data = []
                            for name, data in results.items():
                                row = {'Model': name.upper()}
                                # Check if data is a dictionary and has 'metrics' key
                                if isinstance(data, dict) and 'metrics' in data:
                                    row.update(data['metrics'])
                                elif isinstance(data, dict):
                                    # If data is dict but no 'metrics' key, update with all data
                                    row.update(data)
                                metrics_data.append(row)
                            
                            metrics_df = pd.DataFrame(metrics_data)
                            st.dataframe(metrics_df.round(4), use_container_width=True)
                            
                            # Safe best model selection with error handling
                            try:
                                best_model = min(results.items(), key=lambda x: x[1]['metrics']['rmse'])
                                rmse = best_model[1]['metrics']['rmse']
                                st.info(f"⭐ Best Model: **{best_model[0].upper()}** with RMSE: {rmse:.4f}")
                            except (KeyError, TypeError) as e:
                                st.warning(f"Could not determine best model: {str(e)}")
                                # Use first model as fallback
                                if results:
                                    best_model = list(results.items())[0]
                                    st.info(f"⭐ Using Model: **{best_model[0].upper()}**")
            else:
                st.warning("No numeric columns for regression")
        
        with tab3:
            st.subheader("Clustering Analysis")
            
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            method = st.selectbox("Clustering Method", ["kmeans", "dbscan"])
            
            if st.button("🔍 Perform Clustering"):
                with st.spinner("Clustering data..."):
                    result = safe_execute(
                        "Clustering",
                        perform_clustering,
                        df,
                        n_clusters,
                        method
                    )
                    
                    if result:
                        st.success(f"✅ Found {result['n_clusters']} clusters")
                        
                        df_with_clusters = df.copy()
                        df_with_clusters['Cluster'] = result['labels']
                        
                        numeric_cols = get_numeric_columns(df)
                        if len(numeric_cols) >= 2:
                            fig = create_scatter_plot(
                                df_with_clusters,
                                numeric_cols[0],
                                numeric_cols[1],
                                'Cluster'
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key="clustering_scatter")

elif st.session_state.current_page == "📝 Text Analytics":
    st.title("📝 Text Analytics")
    
    # FIX: Always get fresh data from session state
    df = st.session_state.get('cleaned_data')
    
    if validate_dataframe(df):
        text_cols = get_categorical_columns(df)
        
        if text_cols:
            selected_col = st.selectbox("Select Text Column", text_cols)
            
            if st.button("🔍 Analyze Text"):
                with st.spinner("Analyzing text..."):
                    results = safe_execute(
                        "Text Analytics",
                        analyze_text_column,
                        df,
                        selected_col
                    )
                    
                    if results:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Text Statistics")
                            stats = results.get('statistics', {})
                            st.metric("Total Texts", stats.get('total_texts', 0))
                            st.metric("Avg Length", f"{stats.get('avg_length', 0):.0f} chars")
                            st.metric("Avg Words", f"{stats.get('avg_words', 0):.1f}")
                        
                        with col2:
                            st.subheader("😊 Sentiment Distribution")
                            if not results.get('sentiment', pd.DataFrame()).empty:
                                sentiment_counts = results['sentiment']['label'].value_counts()
                                fig = create_bar_chart(
                                    pd.DataFrame({'Sentiment': sentiment_counts.index, 'Count': sentiment_counts.values}),
                                    'Sentiment',
                                    'Count'
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True, key="text_sentiment_bar")
                        
                        if results.get('wordcloud'):
                            st.subheader("☁️ Word Cloud")
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(results['wordcloud'], interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
                        
                        if results.get('bigrams'):
                            st.subheader("📝 Top Bigrams")
                            bigrams_df = pd.DataFrame(results['bigrams'], columns=['Bigram', 'Frequency'])
                            st.dataframe(bigrams_df, use_container_width=True)
        else:
            st.warning("No text columns found in dataset")

elif st.session_state.current_page == "🚀 Advanced ML":
    st.title("🚀 Advanced ML Models")
    
    if validate_dataframe(df):
        try:
            from modules.advanced_ml import train_neural_network, train_xgboost, train_lightgbm, ensemble_models
            from sklearn.model_selection import train_test_split
        except ImportError as e:
            st.error(f"Advanced ML module not available: {str(e)}")
            st.stop()
        
        tab1, tab2, tab3 = st.tabs(["🧠 Neural Networks", "⚡ XGBoost/LightGBM", "🤝 Ensemble"])
        
        with tab1:
            st.subheader("Multi-Layer Perceptron (Neural Network)")
            
            task = st.radio("Task Type", ["classification", "regression"])
            
            numeric_cols = get_numeric_columns(df)
            categorical_cols = get_categorical_columns(df)
            
            if numeric_cols:
                target_col = st.selectbox("Target Column", df.columns.tolist())
                
                col1, col2 = st.columns(2)
                with col1:
                    hidden_layers = st.text_input("Hidden Layers (comma-separated)", "100,50")
                    activation = st.selectbox("Activation", ["relu", "tanh", "logistic"])
                
                with col2:
                    max_iter = st.number_input("Max Iterations", 100, 2000, 500)
                    test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
                
                if st.button("🚀 Train Neural Network"):
                    with st.spinner("Training neural network..."):
                        try:
                            layers = tuple(int(x.strip()) for x in hidden_layers.split(','))
                            
                            X = df.drop(columns=[target_col])
                            y = df[target_col]
                            
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                            
                            result = train_neural_network(X_train, y_train, X_test, y_test, task, layers, activation, max_iter)
                            
                            if result:
                                st.success("✅ Model trained successfully!")
                                
                                metrics = result['metrics']
                                st.subheader("📊 Model Performance")
                                
                                cols = st.columns(len(metrics))
                                for i, (key, value) in enumerate(metrics.items()):
                                    with cols[i]:
                                        st.metric(key.replace('_', ' ').title(), f"{value:.4f}")
                                
                                st.session_state.trained_models['neural_network'] = result['model']
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        with tab2:
            st.subheader("Gradient Boosting Models")
            
            model_type = st.selectbox("Model Type", ["XGBoost", "LightGBM"])
            task = st.radio("Task", ["classification", "regression"], key="gb_task")
            
            if numeric_cols:
                target_col = st.selectbox("Target Column", df.columns.tolist(), key="gb_target")
                
                col1, col2 = st.columns(2)
                with col1:
                    max_depth = st.slider("Max Depth", 3, 15, 6)
                    n_estimators = st.slider("N Estimators", 50, 500, 100)
                
                with col2:
                    learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
                    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, key="gb_test")
                
                params = {
                    'max_depth': max_depth,
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'random_state': 42
                }
                
                if st.button("⚡ Train Model"):
                    with st.spinner(f"Training {model_type}..."):
                        try:
                            X = df.drop(columns=[target_col])
                            y = df[target_col]
                            
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                            
                            if model_type == "XGBoost":
                                result = train_xgboost(X_train, y_train, X_test, y_test, task, params)
                            else:
                                params['verbosity'] = -1
                                result = train_lightgbm(X_train, y_train, X_test, y_test, task, params)
                            
                            if result:
                                st.success("✅ Model trained successfully!")
                                
                                metrics = result['metrics']
                                st.subheader("📊 Model Performance")
                                
                                cols = st.columns(len(metrics))
                                for i, (key, value) in enumerate(metrics.items()):
                                    with cols[i]:
                                        st.metric(key.replace('_', ' ').title(), f"{value:.4f}")
                                
                                st.subheader("🎯 Feature Importance")
                                st.dataframe(result['feature_importance'].head(10), use_container_width=True)
                                
                                st.session_state.trained_models[model_type.lower()] = result['model']
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        with tab3:
            st.subheader("Ensemble Methods")
            st.info("Train multiple models first, then create an ensemble!")
            
            if st.session_state.trained_models:
                st.success(f"✅ {len(st.session_state.trained_models)} models available for ensembling")
                st.write(f"Models: {', '.join(st.session_state.trained_models.keys())}")
            else:
                st.warning("⚠️ No trained models yet. Train models in tabs above first.")

elif st.session_state.current_page == "⏰ Time Series":
    st.title("⏰ Time Series Analysis")
    
    if validate_dataframe(df):
        try:
            from modules.time_series import (
                fit_arima_model, fit_sarima_model, decompose_time_series,
                check_stationarity, auto_arima, plot_acf_pacf
            )
        except ImportError as e:
            st.error(f"Time Series module not available: {str(e)}")
            st.stop()
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Decomposition", "📈 ARIMA", "🔮 SARIMA", "🎯 Auto ARIMA"])
        
        numeric_cols = get_numeric_columns(df)
        
        with tab1:
            st.subheader("Seasonal Decomposition")
            
            if numeric_cols:
                ts_col = st.selectbox("Select Time Series Column", numeric_cols)
                period = st.number_input("Period (e.g., 12 for monthly)", 2, 365, 12)
                model_type = st.selectbox("Model Type", ["additive", "multiplicative"])
                
                if st.button("🔍 Decompose"):
                    with st.spinner("Decomposing time series..."):
                        result = decompose_time_series(df[ts_col], period, model_type)
                        
                        if result and 'error' not in result:
                            st.success("✅ Decomposition complete!")
                            
                            import matplotlib.pyplot as plt
                            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
                            
                            result['observed'].plot(ax=axes[0], title='Observed')
                            result['trend'].plot(ax=axes[1], title='Trend')
                            result['seasonal'].plot(ax=axes[2], title='Seasonal')
                            result['residual'].plot(ax=axes[3], title='Residual')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        elif result and 'error' in result:
                            st.error(result['error'])
        
        with tab2:
            st.subheader("ARIMA Forecasting")
            
            if numeric_cols:
                ts_col = st.selectbox("Select Time Series Column", numeric_cols, key="arima_col")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    p = st.number_input("p (AR order)", 0, 5, 1)
                with col2:
                    d = st.number_input("d (Differencing)", 0, 2, 1)
                with col3:
                    q = st.number_input("q (MA order)", 0, 5, 1)
                with col4:
                    forecast_steps = st.number_input("Forecast Steps", 1, 100, 10)
                
                if st.button("📈 Fit ARIMA"):
                    with st.spinner("Fitting ARIMA model..."):
                        result = fit_arima_model(df[ts_col], (p, d, q), forecast_steps)
                        
                        if result:
                            st.success(f"✅ ARIMA({p},{d},{q}) fitted successfully!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("AIC", f"{result['aic']:.2f}")
                            with col2:
                                st.metric("BIC", f"{result['bic']:.2f}")
                            
                            st.subheader("🔮 Forecast")
                            forecast_df = pd.DataFrame({
                                'Step': range(1, forecast_steps + 1),
                                'Forecast': result['forecast']
                            })
                            st.dataframe(forecast_df, use_container_width=True)
                            
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax.plot(df[ts_col].values[-50:], label='Historical', marker='o')
                            ax.plot(range(len(df[ts_col])-1, len(df[ts_col])-1+forecast_steps),
                                   result['forecast'], label='Forecast', marker='s', linestyle='--')
                            ax.legend()
                            ax.set_title('ARIMA Forecast')
                            st.pyplot(fig)
        
        with tab3:
            st.subheader("SARIMA (Seasonal ARIMA)")
            
            if numeric_cols:
                ts_col = st.selectbox("Select Time Series Column", numeric_cols, key="sarima_col")
                
                st.markdown("**Non-seasonal parameters**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    p = st.number_input("p", 0, 5, 1, key="sarima_p")
                with col2:
                    d = st.number_input("d", 0, 2, 1, key="sarima_d")
                with col3:
                    q = st.number_input("q", 0, 5, 1, key="sarima_q")
                
                st.markdown("**Seasonal parameters**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    P = st.number_input("P", 0, 3, 1)
                with col2:
                    D = st.number_input("D", 0, 2, 1)
                with col3:
                    Q = st.number_input("Q", 0, 3, 1)
                with col4:
                    s = st.number_input("s (period)", 2, 365, 12)
                
                forecast_steps = st.number_input("Forecast Steps", 1, 100, 10, key="sarima_steps")
                
                if st.button("📈 Fit SARIMA"):
                    with st.spinner("Fitting SARIMA model..."):
                        result = fit_sarima_model(df[ts_col], (p, d, q), (P, D, Q, s), forecast_steps)
                        
                        if result:
                            st.success(f"✅ SARIMA({p},{d},{q})x({P},{D},{Q},{s}) fitted!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("AIC", f"{result['aic']:.2f}")
                            with col2:
                                st.metric("BIC", f"{result['bic']:.2f}")
                            
                            st.subheader("🔮 Forecast")
                            forecast_df = pd.DataFrame({
                                'Step': range(1, forecast_steps + 1),
                                'Forecast': result['forecast']
                            })
                            st.dataframe(forecast_df, use_container_width=True)
        
        with tab4:
            st.subheader("Auto ARIMA - Automatic Parameter Selection")
            
            if numeric_cols:
                ts_col = st.selectbox("Select Time Series Column", numeric_cols, key="auto_col")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    max_p = st.number_input("Max p", 1, 5, 3)
                with col2:
                    max_d = st.number_input("Max d", 1, 2, 2)
                with col3:
                    max_q = st.number_input("Max q", 1, 5, 3)
                
                if st.button("🎯 Find Best ARIMA"):
                    with st.spinner("Searching for best parameters... This may take a while."):
                        result = auto_arima(df[ts_col], max_p, max_d, max_q)
                        
                        if result:
                            st.success(f"✅ Best model found: ARIMA{result['best_order']}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Best Order (p,d,q)", str(result['best_order']))
                                st.metric("AIC", f"{result['aic']:.2f}")
                            with col2:
                                st.metric("BIC", f"{result['bic']:.2f}")
                            
                            st.subheader("🔮 10-Step Forecast")
                            forecast_df = pd.DataFrame({
                                'Step': range(1, 11),
                                'Forecast': result['forecast']
                            })
                            st.dataframe(forecast_df, use_container_width=True)

elif st.session_state.current_page == "📥 Export Center":
    st.title("📥 Export Center")
    try:
        # FIX: Always get fresh data from session state
        df = st.session_state.get('cleaned_data')
        create_export_center(df, st.session_state.get('trained_models'))
    except Exception as e:
        st.error(f"Error in export center: {str(e)}")

elif st.session_state.current_page == "🧠 Deep Learning":
    # FIX: Always get fresh data from session state
    df = st.session_state.get('cleaned_data')
    
    if df is not None and not df.empty:
        all_cols = df.columns.tolist()
        if all_cols:
            target_col = st.selectbox("Select Target Column", all_cols, key="dl_target")
            try:
                DeepLearningModels.create_deep_learning_dashboard(df, target_col)
            except Exception as e:
                st.error(f"Error creating deep learning dashboard: {str(e)}")
        else:
            st.warning("⚠️ No columns available for deep learning")
    else:
        st.warning("⚠️ Please upload data first!")

elif st.session_state.current_page == "📈 Advanced TS":
    # FIX: Always get fresh data from session state
    df = st.session_state.get('cleaned_data')
    
    if df is not None and not df.empty:
        try:
            AdvancedTimeSeriesModels.create_time_series_dashboard(df)
        except Exception as e:
            st.error(f"Error creating time series dashboard: {str(e)}")
    else:
        st.warning("⚠️ Please upload data first!")

elif st.session_state.current_page == "💾 Projects":
    st.title("💾 Save & Load Projects")
    
    # FIX: Always get fresh data from session state
    df = st.session_state.get('cleaned_data')
    
    if not AuthManager.is_authenticated():
        st.warning("⚠️ Please login to save and load projects")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["💾 Save Project", "📂 Load Project", "📊 My Datasets", "🤖 My Models"])
        
        with tab1:
            st.subheader("💾 Save Current Project")
            
            project_name = st.text_input("Project Name")
            project_desc = st.text_area("Description (optional)")
            is_public = st.checkbox("Make Public")
            
            if st.button("💾 Save Project", type="primary"):
                if project_name:
                    # Use existing session_manager instance
                    project_id = session_manager.save_project(project_name, project_desc, is_public)
                    if project_id:
                        st.success(f"✅ Project saved! ID: {project_id}")
                        
                        if df is not None:
                            dataset_id = session_manager.save_dataset(
                                project_id,
                                f"{project_name}_dataset",
                                df
                            )
                            if dataset_id:
                                st.success(f"✅ Dataset saved! ID: {dataset_id}")
                        
                        if st.session_state.get('trained_models'):
                            for model_name, model_data in st.session_state['trained_models'].items():
                                if 'model' in model_data:
                                    model_id = session_manager.save_model(
                                        project_id,
                                        model_name,
                                        model_data['model'],
                                        'classification',
                                        model_name,
                                        model_data.get('metrics', {})
                                    )
                                    if model_id:
                                        st.success(f"✅ Model '{model_name}' saved! ID: {model_id}")
                else:
                    st.warning("⚠️ Please enter project name")
        
        with tab2:
            st.subheader("📂 Load Projects")
            # Use existing session_manager instance
            projects = session_manager.list_user_projects()
            
            if projects:
                for proj in projects:
                    with st.expander(f"📁 {proj['name']}"):
                        st.write(f"**Description:** {proj['description']}")
                        st.write(f"**Created:** {proj['created_at']}")
                        st.write(f"**Updated:** {proj['updated_at']}")
            else:
                st.info("ℹ️ No saved projects yet")
        
        with tab3:
            st.subheader("📊 My Datasets")
            # Use existing session_manager instance
            datasets = session_manager.list_user_datasets()
            
            if datasets:
                for ds in datasets:
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.write(f"**{ds['name']}**")
                    with col2:
                        st.write(f"{ds['rows']} rows × {ds['columns']} cols")
                    with col3:
                        if st.button("📥 Load", key=f"load_ds_{ds['id']}"):
                            loaded_df = session_manager.load_dataset(ds['id'])
                            if loaded_df is not None:
                                st.session_state.uploaded_data = loaded_df
                                st.session_state.cleaned_data = loaded_df.copy()
                                st.success("✅ Dataset loaded!")
                                st.rerun()
            else:
                st.info("ℹ️ No saved datasets yet")
        
        with tab4:
            st.subheader("🤖 My Models")
            # Use existing session_manager instance
            models = session_manager.list_user_models()
            
            if models:
                for mdl in models:
                    with st.expander(f"🤖 {mdl['name']}"):
                        st.write(f"**Algorithm:** {mdl['algorithm']}")
                        st.write(f"**Type:** {mdl['model_type']}")
                        st.write(f"**Metrics:** {mdl['metrics']}")
                        st.write(f"**Created:** {mdl['created_at']}")
            else:
                st.info("ℹ️ No saved models yet")

# Visualizations page
def visualizations_page():
    """Create visualizations"""
    st.markdown("## 📊 Visualizations")
    
    if st.session_state.get('data') is None:
        st.warning("Please upload data first!")
        return
    
    df = st.session_state.data
    
    # Initialize visualizer with caching
    try:
        @st.cache_resource
        def get_visualizer():
            from modules.visualizations import AdvancedVisualizer
            return AdvancedVisualizer()
        
        visualizer = get_visualizer()
    except Exception as e:
        st.error(f"Error initializing visualizer: {str(e)}")
        visualizer = None
    
    # Chart type selection
    chart_type = st.selectbox(
        "Select Chart Type:",
        ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Heatmap", "3D Plot"]
    )
    
    if chart_type in ["Scatter Plot", "Line Chart"]:
        x_col = st.selectbox("X-axis:", df.columns)
        y_col = st.selectbox("Y-axis:", df.columns)
        color_col = st.selectbox("Color by (optional):", [None] + list(df.columns))
        
        if st.button("Generate Chart"):
            try:
                import plotly.express as px
                
                # Create chart with caching for better performance
                @st.cache_data(ttl=300)
                def create_chart(_df, _chart_type, _x_col, _y_col, _color_col):
                    if _chart_type == "Scatter Plot":
                        return px.scatter(_df, x=_x_col, y=_y_col, color=_color_col)
                    else:  # Line Chart
                        return px.line(_df, x=_x_col, y=_y_col, color=_color_col)
                
                fig = create_chart(df, chart_type, x_col, y_col, color_col)
                st.plotly_chart(fig, use_container_width=True, key=f"viz_{chart_type}_{x_col}_{y_col}")
            
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
    
    elif chart_type == "Bar Chart":
        x_col = st.selectbox("Category:", df.select_dtypes(include=['object']).columns)
        y_col = st.selectbox("Value:", df.select_dtypes(include=[np.number]).columns)
        
        if st.button("Generate Chart"):
            try:
                import plotly.express as px
                
                @st.cache_data(ttl=300)
                def create_bar_chart(_df, _x_col, _y_col):
                    return px.bar(_df, x=_x_col, y=_y_col)
                
                fig = create_bar_chart(df, x_col, y_col)
                st.plotly_chart(fig, use_container_width=True, key=f"viz_bar_{x_col}_{y_col}")
            
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
    
    elif chart_type == "Histogram":
        col = st.selectbox("Select column:", df.select_dtypes(include=[np.number]).columns)
        
        if st.button("Generate Chart"):
            try:
                import plotly.express as px
                
                @st.cache_data(ttl=300)
                def create_histogram(_df, _col):
                    return px.histogram(_df, x=_col, marginal="box")
                
                fig = create_histogram(df, col)
                st.plotly_chart(fig, use_container_width=True, key=f"viz_histogram_{col}")
            
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
    
    elif chart_type == "Box Plot":
        y_col = st.selectbox("Value:", df.select_dtypes(include=[np.number]).columns)
        x_col = st.selectbox("Category (optional):", [None] + list(df.select_dtypes(include=['object']).columns))
        
        if st.button("Generate Chart"):
            try:
                import plotly.express as px
                
                @st.cache_data(ttl=300)
                def create_box_plot(_df, _y_col, _x_col):
                    return px.box(_df, y=_y_col, x=_x_col)
                
                fig = create_box_plot(df, y_col, x_col)
                st.plotly_chart(fig, use_container_width=True, key=f"viz_box_{y_col}_{x_col}")
            
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
    
    elif chart_type == "Heatmap":
        if st.button("Generate Chart"):
            try:
                # Use cached correlation matrix
                numeric_cols = get_numeric_columns(df)
                if len(numeric_cols) >= 2:
                    corr_matrix = df[numeric_cols].corr()
                    import plotly.express as px
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
                    st.plotly_chart(fig, use_container_width=True, key="viz_heatmap")
                else:
                    st.warning("Need at least 2 numeric columns for heatmap")
            
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")

# Text Analytics page
def text_analytics_page():
    """Text analytics functionality"""
    st.markdown("## 📝 Text Analytics")
    
    if st.session_state.get('data') is None:
        st.warning("Please upload data first!")
        return
    
    df = st.session_state.data
    
    # Select text column
    text_cols = df.select_dtypes(include=['object']).columns
    if len(text_cols) == 0:
        st.warning("No text columns found in the data!")
        return
    
    text_col = st.selectbox("Select text column:", text_cols)
    
    # Analytics options
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        ["Sentiment Analysis", "Word Cloud", "Text Statistics", "Topic Modeling"]
    )
    
    if st.button("Run Analysis"):
        try:
            with st.spinner("Running text analysis..."):
                if analysis_type == "Sentiment Analysis":
                    # Simple sentiment analysis with caching
                    @st.cache_data(ttl=600)
                    def analyze_sentiment(_texts):
                        from textblob import TextBlob
                        sentiments = []
                        for text in _texts.dropna():
                            blob = TextBlob(str(text))
                            sentiments.append(blob.sentiment.polarity)
                        return sentiments
                    
                    sentiments = analyze_sentiment(df[text_col])
                    df_sentiment = df.copy()
                    df_sentiment['sentiment'] = sentiments
                    
                    st.markdown("### 😊 Sentiment Analysis Results")
                    
                    # Overall sentiment
                    avg_sentiment = np.mean(sentiments)
                    if avg_sentiment > 0.1:
                        sentiment_label = "Positive"
                        color = "green"
                    elif avg_sentiment < -0.1:
                        sentiment_label = "Negative"
                        color = "red"
                    else:
                        sentiment_label = "Neutral"
                        color = "gray"
                    
                    st.markdown(f"**Overall Sentiment:** <span style='color: {color}'>{sentiment_label}</span>", unsafe_allow_html=True)
                    st.metric("Average Sentiment Score", f"{avg_sentiment:.3f}")
                    
                    # Sentiment distribution
                    import plotly.express as px
                    fig = px.histogram(df_sentiment, x='sentiment', nbins=20, title="Sentiment Distribution")
                    st.plotly_chart(fig, use_container_width=True, key="text_sentiment_histogram")
                
                elif analysis_type == "Word Cloud":
                    from wordcloud import WordCloud
                    import matplotlib.pyplot as plt
                    
                    # Combine all text with caching
                    @st.cache_data(ttl=600)
                    def generate_wordcloud(_text_data):
                        # Generate word cloud
                        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(_text_data)
                        return wordcloud
                    
                    text_data = ' '.join(df[text_col].dropna().astype(str))
                    wordcloud = generate_wordcloud(text_data)
                    
                    st.markdown("### ☁️ Word Cloud")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
                
                elif analysis_type == "Text Statistics":
                    text_data = df[text_col].dropna().astype(str)
                    
                    # Calculate statistics
                    char_counts = [len(text) for text in text_data]
                    word_counts = [len(text.split()) for text in text_data]
                    
                    st.markdown("### 📊 Text Statistics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Texts", len(text_data))
                        st.metric("Avg Characters", f"{np.mean(char_counts):.1f}")
                        st.metric("Max Characters", max(char_counts))
                    
                    with col2:
                        st.metric("Total Words", sum(word_counts))
                        st.metric("Avg Words", f"{np.mean(word_counts):.1f}")
                        st.metric("Max Words", max(word_counts))
                    
                    # Visualizations
                    import plotly.express as px
                    
                    fig1 = px.histogram(x=char_counts, nbins=20, title="Character Count Distribution")
                    st.plotly_chart(fig1, use_container_width=True, key="text_char_histogram")
                    
                    fig2 = px.histogram(x=word_counts, nbins=20, title="Word Count Distribution")
                    st.plotly_chart(fig2, use_container_width=True, key="text_word_histogram")
                
                elif analysis_type == "Topic Modeling":
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.decomposition import LatentDirichletAllocation
                    
                    # Prepare text data
                    text_data = df[text_col].dropna().astype(str)
                    
                    # LDA with caching
                    @st.cache_data(ttl=1200)  # Cache for 20 minutes
                    def perform_topic_modeling(_text_data, _n_topics):
                        # TF-IDF Vectorization
                        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                        tfidf_matrix = vectorizer.fit_transform(_text_data)
                        
                        # LDA
                        lda = LatentDirichletAllocation(n_components=_n_topics, random_state=42)
                        lda.fit(tfidf_matrix)
                        
                        return vectorizer, lda
                    
                    n_topics = st.slider("Number of topics:", 2, 10, 5)
                    vectorizer, lda = perform_topic_modeling(text_data, n_topics)
                    
                    # Display topics
                    st.markdown("### 🎯 Topics")
                    
                    feature_names = vectorizer.get_feature_names_out()
                    for topic_idx, topic in enumerate(lda.components_):
                        top_words_idx = topic.argsort()[-10:][::-1]
                        top_words = [feature_names[i] for i in top_words_idx]
                        st.write(f"**Topic {topic_idx + 1}:** {' '.join(top_words)}")
        
        except Exception as e:
            st.error(f"Error during text analysis: {str(e)}")

# AI Assistant page
def ai_assistant_page():
    """AI Chat Assistant"""
    st.markdown("## 🤖 AI Assistant")
    
    # Initialize Gemini assistant
    if 'gemini_assistant' not in st.session_state or st.session_state.gemini_assistant is None:
        # Create a simple placeholder for gemini_assistant if initialize_components is not available
        st.session_state.gemini_assistant = None
    
    # Chat interface
    st.markdown("### 💬 Chat with AI Assistant")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**AI Assistant:** {message['content']}")
    
    # Input field
    user_input = st.text_input("Type your message:", key="user_input")
    
    if st.button("Send") and user_input:
        try:
            with st.spinner("AI is thinking..."):
                # Add user message to history
                st.session_state.chat_history.append({'role': 'user', 'content': user_input})
                
                # Get AI response
                if st.session_state.gemini_assistant:
                    response = st.session_state.gemini_assistant.chat(user_input)
                    st.session_state.chat_history.append({'role': 'assistant', 'content': response.get('response', 'No response generated')})
                else:
                    response = "AI Assistant is not available. Please check your API key."
                    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                
                st.rerun()
        
        except Exception as e:
            st.error(f"Error getting AI response: {str(e)}")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    quick_actions = [
        "Analyze my data",
        "Suggest visualizations",
        "Explain machine learning results",
        "Help with data cleaning"
    ]
    
    for action in quick_actions:
        if st.button(action, key=f"quick_{action}"):
            st.session_state.chat_history.append({'role': 'user', 'content': action})
            st.rerun()

# Export page
def export_page():
    """Export functionality"""
    st.markdown("## 📤 Export")
    
    if st.session_state.get('data') is None:
        st.warning("Please upload data first!")
        return
    
    df = st.session_state.data
    
    # Export options
    export_format = st.selectbox(
        "Select Export Format:",
        ["CSV", "Excel", "JSON", "PDF Report", "Jupyter Notebook"]
    )
    
    if export_format in ["CSV", "Excel", "JSON"]:
        filename = st.text_input("Filename:", value="exported_data")
        
        if st.button("Export"):
            try:
                if export_format == "CSV":
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{filename}.csv",
                        mime="text/csv"
                    )
                
                elif export_format == "Excel":
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Data')
                    st.download_button(
                        label="Download Excel",
                        data=output.getvalue(),
                        file_name=f"{filename}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                elif export_format == "JSON":
                    json_data = df.to_json(orient='records')
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"{filename}.json",
                        mime="application/json"
                    )
            
            except Exception as e:
                st.error(f"Error exporting data: {str(e)}")
    
    elif export_format == "PDF Report":
        st.info("PDF export functionality requires additional setup. Please check documentation.")
    
    elif export_format == "Jupyter Notebook":
        st.info("Jupyter notebook export generates a template for your analysis.")
        
        if st.button("Generate Notebook"):
            try:
                # Generate notebook template
                notebook_template = {
                    "cells": [
                        {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": [
                                "# Data Analysis Report\n",
                                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                                f"Dataset shape: {df.shape}"
                            ]
                        },
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": [
                                "import pandas as pd\n",
                                "import numpy as np\n",
                                "import matplotlib.pyplot as plt\n",
                                "import seaborn as sns\n",
                                "\n",
                                "# Load data\n",
                                "df = pd.read_csv('your_data.csv')\n",
                                "print(df.head())\n",
                                "print(df.info())"
                            ]
                        }
                    ],
                    "metadata": {
                        "kernelspec": {
                            "display_name": "Python 3",
                            "language": "python",
                            "name": "python3"
                        }
                    },
                    "nbformat": 4,
                    "nbformat_minor": 4
                }
                
                import json
                notebook_json = json.dumps(notebook_template, indent=2)
                
                st.download_button(
                    label="Download Notebook",
                    data=notebook_json,
                    file_name="analysis_template.ipynb",
                    mime="application/json"
                )
            
            except Exception as e:
                st.error(f"Error generating notebook: {str(e)}")

# Collaboration page
def collaboration_page():
    """Collaboration features"""
    st.markdown("## 🤝 Collaboration")
    
    st.info("Collaboration features require backend setup. Please check deployment guide.")
    
    # Placeholder for collaboration features
    st.markdown("### 📋 Features Coming Soon:")
    
    features = [
        "Real-time collaboration",
        "Version control",
        "Team workspaces",
        "Comment and annotation system",
        "Sharing and permissions"
    ]
    
    for feature in features:
        st.write(f"- {feature}")

# Settings page
def settings_page():
    """Settings page"""
    st.markdown("## ⚙️ Settings")
    
    # API Key settings
    st.markdown("### 🔑 API Keys")
    
    gemini_api_key = st.text_input(
        "Gemini API Key:",
        type="password",
        value=os.getenv('GEMINI_API_KEY', ''),
        help="Enter your Google Gemini API key"
    )
    
    if st.button("Save API Key"):
        os.environ['GEMINI_API_KEY'] = gemini_api_key
        st.success("API Key saved!")
    
    # Display settings
    st.markdown("### 🎨 Display Settings")
    
    theme = st.selectbox(
        "Theme:",
        ["Light", "Dark", "Auto"],
        index=2
    )
    
    # Performance settings
    st.markdown("### ⚡ Performance Settings")
    
    max_rows = st.slider(
        "Maximum rows to display:",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )
    
    # Cache settings
    st.markdown("### 💾 Cache Settings")
    
    if st.button("Clear Cache"):
        try:
            st.cache_resource.clear()
        except:
            # Fallback for older Streamlit versions
            st.experimental_singleton.clear()
            st.experimental_memo.clear()
        st.success("Cache cleared!")
    
    # About section
    st.markdown("### ℹ️ About")
    st.markdown("""
    **ProjekData v1.0.0**
    
    AI Data Analysis Platform powered by:
    - Google Gemini 2.5 Flash
    - Streamlit
    - scikit-learn
    - Plotly
    - pandas
    
    © 2024 ProjekData Team
    """)

# Main app logic
def main():
    """Main application logic"""
    # Initialize components
    if 'pipeline' not in st.session_state or 'ml_models' not in st.session_state:
        st.session_state.pipeline = None
        st.session_state.ml_models = {}
        st.session_state.trained_models = {}
    
    # Main content is already handled above with the page navigation
    # No need for additional page handling here
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "🚀 ProjekData - AI Data Analysis Platform | Made with ❤️ in Indonesia"
        "</div>",
        unsafe_allow_html=True
    )

# Error handling
def handle_app_error():
    """Handle application errors"""
    try:
        error = st.query_params.get('error')
        if error:
            st.error(f"Error: {error}")
    except:
        # Fallback for older Streamlit versions
        try:
            error = st.experimental_get_query_params().get('error')
            if error:
                st.error(f"Error: {error}")
        except:
            pass

# Run the app
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please check the troubleshooting guide or contact support.")
        # Log error for debugging
        print(f"Error: {str(e)}")
        try:
            import traceback
            print(traceback.format_exc())
        except ImportError:
            print("Traceback module not available")

st.markdown("---")

st.subheader("💬 AI Chat Assistant")

show_rate_limit_status()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history[-10:]:
        role = message.get("role", "user")
        content = message.get("content", "")
        
        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**AI:** {content}")

can_request, wait_time = can_make_request()

if not can_request and wait_time != "hourly_limit":
    st.warning(f"⏳ Please wait {wait_time} seconds before asking next question")
elif not can_request and wait_time == "hourly_limit":
    st.error(f"🚫 Hourly limit reached! ({get_remaining_requests()}/15 requests remaining)")

user_input = st.text_input(
    "Ask AI about your data",
    placeholder="e.g., Show correlation between age and income",
    disabled=not can_request,
    key="chat_input"
)

if st.button("📤 Send", disabled=not can_request, type="primary"):
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("🤖 AI is thinking..."):
            try:
                # FIX: Always get fresh data from session state for AI context
                current_df = st.session_state.get('cleaned_data')
                context = get_data_context(current_df) if current_df is not None else {}
                response = chat_with_gemini(
                    user_input,
                    st.session_state.chat_history,
                    context
                )
            except Exception as e:
                st.error(f"Error getting AI response: {str(e)}")
                response = "Sorry, I encountered an error while processing your request. Please try again."
            
            if response:
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                update_rate_limit()
                st.rerun()

st.markdown("---")
st.caption("AI Data Analysis Platform by Muhammad Irbabul Salas | Powered by Gemini 2.5 Flash")