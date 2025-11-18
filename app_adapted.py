import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

from modules.data_processing import load_data_from_file, profile_data, clean_data
from modules.ml_models import train_classification_models, train_regression_models, perform_clustering, perform_pca, get_feature_importance
from modules.text_analytics import analyze_text_column, analyze_sentiment
from modules.visualizations import (
    create_scatter_plot, create_bar_chart, create_histogram, create_box_plot,
    create_correlation_heatmap, create_line_chart, create_violin_plot,
    create_confusion_matrix_plot, create_feature_importance_plot
)
from modules.gemini_integration import chat_with_gemini, get_data_context
from modules.glm_integration import chat_with_glm, generate_glm_insights, is_glm_available, get_glm_status
from modules.openai_integration import chat_with_openai, generate_openai_insights, is_openai_available, get_openai_status
from modules.export_handler import create_export_center
from utils.helpers import initialize_session_state, get_numeric_columns, get_categorical_columns, calculate_data_quality_score
from utils.rate_limiter import initialize_rate_limiter, can_make_request, show_rate_limit_status, update_rate_limit, get_remaining_requests
from utils.error_handler import safe_execute, validate_dataframe
from database.auth_manager import auth_manager, AuthManager
from database.session_manager import session_manager, SessionManager

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
        else:
            st.image("https://via.placeholder.com/80", width=80)
    
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
            st.session_state.uploaded_data = df
            st.session_state.cleaned_data = df.copy()
    
    if st.button("📝 Load Sample E-commerce Data"):
        try:
            df = pd.read_csv("assets/sample_datasets/sample_ecommerce.csv")
            st.session_state.uploaded_data = df
            st.session_state.cleaned_data = df.copy()
            st.success("✅ Sample data loaded!")
            st.rerun()
        except:
            st.error("❌ Sample data not found")
    
    if st.button("💬 Load Sample Reviews Data"):
        try:
            df = pd.read_csv("assets/sample_datasets/sample_reviews.csv")
            st.session_state.uploaded_data = df
            st.session_state.cleaned_data = df.copy()
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
                st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("📈 Data Preview")
            st.dataframe(df.head(10), width='stretch')
            st.caption(f"Showing 10 of {len(df):,} rows")
        
        st.markdown("---")
        st.subheader("🤖 AI Quick Insights")
        
        # AI Model Selection
        col1, col2 = st.columns([2, 1])
        with col1:
            ai_model = st.selectbox(
                "🤖 Pilih AI Model",
                ["Gemini 2.5 Flash", "GLM-4.6", "GPT-4o-mini"],
                help="Pilih model AI untuk generate insights"
            )
        with col2:
            if ai_model == "GLM-4.6":
                glm_status = get_glm_status()
                if glm_status['configured']:
                    st.success("✅ GLM-4.6 Ready")
                else:
                    st.warning("⚠️ GLM API Key Needed")
            elif ai_model == "GPT-4o-mini":
                openai_status = get_openai_status()
                if openai_status['configured']:
                    st.success("✅ GPT-4o-mini Ready")
                else:
                    st.warning("⚠️ OpenAI API Key Needed")
        
        if st.button("✨ Generate Insights with AI"):
            with st.spinner("Analyzing your data..."):
                if ai_model == "GLM-4.6":
                    insights = safe_execute("GLM-4.6 Insights", generate_glm_insights, df, "general")
                elif ai_model == "GPT-4o-mini":
                    insights = safe_execute("GPT-4o-mini Insights", generate_openai_insights, df, "general")
                else:
                    from modules.gemini_integration import generate_insights
                    insights = safe_execute("Gemini Insights", generate_insights, df, "general")
                
                if insights:
                    st.success(insights)
    else:
        st.info("📤 Please upload a dataset or load sample data from the sidebar to begin analysis.")

elif st.session_state.current_page == "🔍 Data Profiling":
    st.title("🔍 Data Profiling")
    
    if df is not None and not df.empty:
        profile = safe_execute("Data Profiling", profile_data, df)
        
        if profile:
            st.subheader("📋 Basic Information")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Rows", f"{profile['basic_info']['rows']:,}")
            with col2:
                st.metric("Columns", profile['basic_info']['columns'])
            with col3:
                st.metric("Duplicates", profile['basic_info']['duplicates'])
            with col4:
                st.metric("Quality", f"{profile['quality_score']}%")
            
            st.markdown("---")
            
            st.subheader("📊 Column Details")
            col_info_df = pd.DataFrame(profile['column_info']).T
            st.dataframe(col_info_df, width='stretch')
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Correlation Heatmap")
                fig = create_correlation_heatmap(df)
                if fig:
                    st.plotly_chart(fig, width='stretch')
            
            with col2:
                st.subheader("🧹 Data Cleaning")
                
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
                
                if st.button("🧹 Clean Data Now"):
                    with st.spinner("Cleaning data..."):
                        cleaned_df, report = safe_execute(
                            "Data Cleaning",
                            clean_data,
                            df,
                            handle_missing,
                            remove_dup,
                            handle_outliers
                        )
                        
                        if cleaned_df is not None:
                            st.session_state.cleaned_data = cleaned_df
                            st.success("✅ Data cleaned successfully!")
                            
                            for action in report.get('actions', []):
                                st.write(f"• {action}")
                            
                            st.rerun()

elif st.session_state.current_page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    
    if df is not None and not df.empty:
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
                        st.plotly_chart(fig, width='stretch')
                
                with col2:
                    st.subheader("📦 Box Plot")
                    fig = create_box_plot(df, selected_col)
                    if fig:
                        st.plotly_chart(fig, width='stretch')
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
                    st.plotly_chart(fig, width='stretch')
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
                    st.plotly_chart(fig, width='stretch')
            else:
                st.warning("Need both categorical and numeric columns")

elif st.session_state.current_page == "🤖 ML Models":
    st.title("🤖 Machine Learning Models")
    
    if df is not None and not df.empty and len(df) >= 50:
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
                            if isinstance(data, dict) and 'metrics' in data:
                                row.update(data['metrics'])
                            metrics_data.append(row)
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df.round(4), width='stretch')
                        
                        # Find best model safely
                        best_model = None
                        best_accuracy = 0
                        for name, data in results.items():
                            if isinstance(data, dict) and 'metrics' in data and isinstance(data['metrics'], dict):
                                accuracy = data['metrics'].get('accuracy', 0)
                                if accuracy > best_accuracy:
                                    best_accuracy = accuracy
                                    best_model = (name, data)
                        
                        if best_model:
                            st.info(f"⭐ Best Model: **{best_model[0].upper()}** with {best_accuracy:.2%} accuracy")
                            
                            if isinstance(best_model[1], dict) and 'confusion_matrix' in best_model[1]:
                                st.subheader("📊 Confusion Matrix (Best Model)")
                                cm_fig = create_confusion_matrix_plot(np.array(best_model[1]['confusion_matrix']))
                                if cm_fig:
                                    st.plotly_chart(cm_fig, width='stretch')
                            
                            numeric_cols = get_numeric_columns(df.drop(columns=[target_col]))
                            importance = None
                            if isinstance(best_model[1], dict) and 'model' in best_model[1]:
                                importance = get_feature_importance(best_model[1]['model'], numeric_cols)
                            
                            if importance:
                                st.subheader("🎯 Feature Importance")
                                fig = create_feature_importance_plot(importance)
                                if fig:
                                    st.plotly_chart(fig, width='stretch')
        
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
                                if isinstance(data, dict) and 'metrics' in data and isinstance(data['metrics'], dict):
                                    row.update(data['metrics'])
                                metrics_data.append(row)
                            
                            if metrics_data:
                                metrics_df = pd.DataFrame(metrics_data)
                                st.dataframe(metrics_df.round(4), width='stretch')
                            
                            # Find best regression model safely
                            best_model = None
                            best_rmse = float('inf')
                            for name, data in results.items():
                                if isinstance(data, dict) and 'metrics' in data and isinstance(data['metrics'], dict):
                                    rmse = data['metrics'].get('rmse', float('inf'))
                                    if rmse < best_rmse:
                                        best_rmse = rmse
                                        best_model = (name, data)
                            
                            if best_model:
                                st.info(f"⭐ Best Model: **{best_model[0].upper()}** with RMSE: {best_rmse:.4f}")
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
                    
                    if result and isinstance(result, dict):
                        n_clusters = result.get('n_clusters', 0)
                        labels = result.get('labels', [])
                        
                        if n_clusters > 0 and len(labels) > 0:
                            st.success(f"✅ Found {n_clusters} clusters")
                            
                            df_with_clusters = df.copy()
                            df_with_clusters['Cluster'] = labels
                            
                            numeric_cols = get_numeric_columns(df)
                            if len(numeric_cols) >= 2:
                                fig = create_scatter_plot(
                                    df_with_clusters,
                                    numeric_cols[0],
                                    numeric_cols[1],
                                    'Cluster'
                                )
                                if fig:
                                    st.plotly_chart(fig, width='stretch')
                        else:
                            st.warning("⚠️ Clustering did not produce valid results")

elif st.session_state.current_page == "📝 Text Analytics":
    st.title("📝 Text Analytics")
    
    if df is not None and not df.empty:
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
                    
                    if results and isinstance(results, dict):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Text Statistics")
                            stats = results.get('statistics', {})
                            if isinstance(stats, dict):
                                st.metric("Total Texts", stats.get('total_texts', 0))
                                st.metric("Avg Length", f"{stats.get('avg_length', 0):.0f} chars")
                                st.metric("Avg Words", f"{stats.get('avg_words', 0):.1f}")
                        
                        with col2:
                            st.subheader("😊 Sentiment Distribution")
                            sentiment_data = results.get('sentiment')
                            if sentiment_data is not None and not sentiment_data.empty:
                                if 'label' in sentiment_data.columns:
                                    sentiment_counts = sentiment_data['label'].value_counts()
                                    fig = create_bar_chart(
                                        pd.DataFrame({'Sentiment': sentiment_counts.index, 'Count': sentiment_counts.values}),
                                        'Sentiment',
                                        'Count'
                                    )
                                    if fig:
                                        st.plotly_chart(fig, width='stretch')
                        
                        if results.get('wordcloud'):
                            st.subheader("☁️ Word Cloud")
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(results['wordcloud'], interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
                        
                        bigrams = results.get('bigrams')
                        if bigrams and isinstance(bigrams, list):
                            st.subheader("📝 Top Bigrams")
                            bigrams_df = pd.DataFrame(bigrams, columns=['Bigram', 'Frequency'])
                            st.dataframe(bigrams_df, width='stretch')
        else:
            st.warning("No text columns found in dataset")

elif st.session_state.current_page == "🚀 Advanced ML":
    st.title("🚀 Advanced ML Models")
    
    if df is not None and not df.empty:
        try:
            from modules.advanced_ml import train_neural_network, train_xgboost, train_lightgbm, ensemble_models
            from sklearn.model_selection import train_test_split
            advanced_ml_available = True
        except ImportError as e:
            st.error(f"⚠️ Advanced ML module not available: {str(e)}")
            st.info("💡 Install required packages: `pip install xgboost lightgbm catboost`")
            advanced_ml_available = False
        
        if not advanced_ml_available:
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
                                st.dataframe(result['feature_importance'].head(10), width='stretch')
                                
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
                st.warning("⚠️ No trained models yet. Train models in the tabs above first.")

elif st.session_state.current_page == "⏰ Time Series":
    st.title("⏰ Time Series Analysis")
    
    if df is not None and not df.empty:
        try:
            from modules.time_series import (
                fit_arima_model, fit_sarima_model, decompose_time_series,
                check_stationarity, auto_arima, plot_acf_pacf
            )
            time_series_available = True
        except ImportError as e:
            st.error(f"⚠️ Time series module not available: {str(e)}")
            st.info("💡 Install required packages: `pip install statsmodels`")
            time_series_available = False
        
        if not time_series_available:
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
                            st.dataframe(forecast_df, width='stretch')
                            
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
                            st.dataframe(forecast_df, width='stretch')
        
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
                            st.dataframe(forecast_df, width='stretch')

elif st.session_state.current_page == "🧠 Deep Learning":
    if df is not None and not df.empty:
        all_cols = df.columns.tolist()
        if all_cols:
            target_col = st.selectbox("Select Target Column", all_cols, key="dl_target")
            try:
                from modules.deep_learning import DeepLearningModels
                DeepLearningModels.create_deep_learning_dashboard(df, target_col)
            except ImportError as e:
                st.error(f"⚠️ Deep learning module not available: {str(e)}")
                st.info("💡 Install required packages: `pip install tensorflow keras`")
        else:
            st.warning("⚠️ No columns available for deep learning")
    else:
        st.warning("⚠️ Please upload data first!")

elif st.session_state.current_page == "📈 Advanced TS":
    if df is not None and not df.empty:
        try:
            from modules.advanced_timeseries import AdvancedTimeSeriesModels
            AdvancedTimeSeriesModels.create_time_series_dashboard(df)
        except ImportError as e:
            st.error(f"⚠️ Advanced time series module not available: {str(e)}")
            st.info("💡 Install required packages: `pip install statsmodels`")
    else:
        st.warning("⚠️ Please upload data first!")

elif st.session_state.current_page == "💾 Projects":
    st.title("💾 Save & Load Projects")
    
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
                    project_id = SessionManager.save_project(project_name, project_desc, is_public)
                    if project_id:
                        st.success(f"✅ Project saved! ID: {project_id}")
                        
                        if df is not None:
                            dataset_id = SessionManager.save_dataset(
                                project_id,
                                f"{project_name}_dataset",
                                df
                            )
                            if dataset_id:
                                st.success(f"✅ Dataset saved! ID: {dataset_id}")
                        
                        if st.session_state.get('trained_models'):
                            for model_name, model_data in st.session_state['trained_models'].items():
                                if 'model' in model_data:
                                    model_id = SessionManager.save_model(
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
            projects = SessionManager.list_user_projects()
            
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
            datasets = SessionManager.list_user_datasets()
            
            if datasets:
                for ds in datasets:
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.write(f"**{ds['name']}**")
                    with col2:
                        st.write(f"{ds['rows']} rows × {ds['columns']} cols")
                    with col3:
                        if st.button("📥 Load", key=f"load_ds_{ds['id']}"):
                            loaded_df = SessionManager.load_dataset(ds['id'])
                            if loaded_df is not None:
                                st.session_state.uploaded_data = loaded_df
                                st.session_state.cleaned_data = loaded_df.copy()
                                st.success("✅ Dataset loaded!")
                                st.rerun()
            else:
                st.info("ℹ️ No saved datasets yet")
        
        with tab4:
            st.subheader("🤖 My Models")
            models = SessionManager.list_user_models()
            
            if models:
                for mdl in models:
                    with st.expander(f"🤖 {mdl['name']}"):
                        st.write(f"**Algorithm:** {mdl['algorithm']}")
                        st.write(f"**Type:** {mdl['model_type']}")
                        st.write(f"**Metrics:** {mdl['metrics']}")
                        st.write(f"**Created:** {mdl['created_at']}")
            else:
                st.info("ℹ️ No saved models yet")

elif st.session_state.current_page == "📥 Export Center":
    st.title("📥 Export Center")
    create_export_center(df, st.session_state.get('trained_models'))

st.markdown("---")

st.subheader("💬 AI Chat Assistant")

# AI Model Selection for Chat
col1, col2 = st.columns([2, 1])
with col1:
    chat_ai_model = st.selectbox(
        "🤖 Pilih AI Model untuk Chat",
        ["Gemini 2.5 Flash", "GLM-4.6", "GPT-4o-mini"],
        key="chat_ai_model",
        help="Pilih model AI untuk chat assistant"
    )
with col2:
    if chat_ai_model == "GLM-4.6":
        glm_status = get_glm_status()
        if glm_status['configured']:
            st.success("✅ GLM-4.6 Ready")
        else:
            st.warning("⚠️ GLM API Key Needed")
    elif chat_ai_model == "GPT-4o-mini":
        openai_status = get_openai_status()
        if openai_status['configured']:
            st.success("✅ GPT-4o-mini Ready")
        else:
            st.warning("⚠️ OpenAI API Key Needed")

# Show rate limit status only for Gemini
if chat_ai_model == "Gemini 2.5 Flash":
    show_rate_limit_status()
else:
    if chat_ai_model == "GLM-4.6":
        st.info("🤖 GLM-4.6: No rate limits (based on your API quota)")
    elif chat_ai_model == "GPT-4o-mini":
        st.info("🤖 GPT-4o-mini: Fast & cost-effective (based on your OpenAI quota)")
    else:
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

# Check if user can make request (only for Gemini)
can_request = True
wait_time = 0
if chat_ai_model == "Gemini 2.5 Flash":
    can_request, wait_time = can_make_request()
    
    if not can_request and wait_time != "hourly_limit":
        st.warning(f"⏳ Please wait {wait_time} seconds before asking next question")
    elif not can_request and wait_time == "hourly_limit":
        st.error(f"🚫 Hourly limit reached! ({get_remaining_requests()}/15 requests remaining)")

user_input = st.text_input(
    "Ask AI about your data",
    placeholder="e.g., Tunjukkan korelasi antara usia dan pendapatan",
    disabled=not can_request if chat_ai_model == "Gemini 2.5 Flash" else False,
    key="chat_input"
)

if st.button("📤 Send", disabled=not can_request if chat_ai_model == "Gemini 2.5 Flash" else False, type="primary"):
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("🤖 AI is thinking..."):
            context = get_data_context(df) if df is not None else {}
            
            if chat_ai_model == "GLM-4 Flash":
                response = chat_with_glm(
                    st.session_state.chat_history[:-1],  # Exclude current user message
                    context,
                    max_tokens=1000
                )
            else:
                response = chat_with_gemini(
                    user_input,
                    st.session_state.chat_history,
                    context
                )
            
            if response:
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                if chat_ai_model == "Gemini 2.5 Flash":
                    update_rate_limit()
                st.rerun()

st.markdown("---")
st.caption("AI Data Analysis Platform by Muhammad Irbabul Salas | Powered by Gemini 2.5 Flash")