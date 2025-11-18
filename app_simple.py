import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Configure Streamlit page
st.set_page_config(
    page_title="🚀 ProjekData - Simple Version",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-message {
        background: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Home", "📁 Data Upload", "📊 Data Analysis", "ℹ️ About"]
)

# Home page
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🚀 ProjekData</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI Data Analysis Platform - Simple Version</p>', unsafe_allow_html=True)
    
    st.markdown("### 🎯 Quick Start")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 1. Upload Your Data")
        st.info("Upload CSV files to get started")
        if st.button("📁 Upload Data Now", key="quick_upload"):
            st.session_state.page = "Data Upload"
            st.rerun()
    
    with col2:
        st.markdown("#### 2. Try Sample Data")
        st.info("Explore with our sample datasets")
        if st.button("📊 Load Sample Data", key="quick_sample"):
            # Load sample data
            try:
                sample_path = "assets/sample_datasets/sample_sales.csv"
                if os.path.exists(sample_path):
                    df = pd.read_csv(sample_path)
                    st.session_state.data = df
                    st.success("Sample data loaded successfully!")
                    st.rerun()
                else:
                    st.error("Sample data not found!")
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
    
    # Display current data info
    if st.session_state.data is not None:
        st.markdown("### 📈 Current Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rows", len(st.session_state.data))
        
        with col2:
            st.metric("Columns", len(st.session_state.data.columns))
        
        with col3:
            st.metric("Memory Usage", f"{st.session_state.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.markdown("#### 🔍 Data Preview")
        st.dataframe(st.session_state.data.head())

# Data Upload page
elif page == "📁 Data Upload":
    st.markdown("## 📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload CSV files for analysis"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            
            # Display data info
            st.markdown("### 📊 Data Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Rows", len(df))
            
            with col2:
                st.metric("Columns", len(df.columns))
            
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Display data preview
            st.markdown("### 🔍 Data Preview")
            st.dataframe(df.head())
            
            # Display data types
            st.markdown("### 📋 Data Types")
            st.write(df.dtypes)
            
            # Display missing values
            st.markdown("### ❓ Missing Values")
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                st.write(missing_data[missing_data > 0])
            else:
                st.success("No missing values found!")
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Sample data section
    st.markdown("### 📊 Sample Datasets")
    
    sample_datasets = {
        "Sales Data": "assets/sample_datasets/sample_sales.csv",
        "Customer Reviews": "assets/sample_datasets/customer_reviews.csv",
        "Iris Dataset": "assets/sample_datasets/iris_dataset.csv",
        "Indonesian News": "assets/sample_datasets/indonesian_news.csv",
        "Financial Data": "assets/sample_datasets/financial_data.csv"
    }
    
    selected_sample = st.selectbox("Choose a sample dataset:", list(sample_datasets.keys()))
    
    if st.button("Load Sample Data"):
        try:
            sample_path = sample_datasets[selected_sample]
            if os.path.exists(sample_path):
                df = pd.read_csv(sample_path)
                st.session_state.data = df
                st.success(f"Sample dataset '{selected_sample}' loaded successfully!")
                st.rerun()
            else:
                st.error("Sample data file not found!")
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")

# Data Analysis page
elif page == "📊 Data Analysis":
    st.markdown("## 📊 Data Analysis")
    
    if st.session_state.data is None:
        st.warning("Please upload data first!")
        st.info("Go to Data Upload page to load your data.")
        st.stop()
    
    df = st.session_state.data
    
    # Analysis options
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        ["Data Overview", "Statistical Summary", "Column Analysis", "Missing Values Analysis"]
    )
    
    if analysis_type == "Data Overview":
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Rows", len(df))
            st.metric("Total Columns", len(df.columns))
            st.metric("Memory Usage (MB)", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}")
        
        with col2:
            st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
            st.metric("Text Columns", len(df.select_dtypes(include=['object']).columns))
            st.metric("Duplicate Rows", df.duplicated().sum())
        
        # Data types
        st.markdown("### 📋 Data Types")
        dtype_counts = df.dtypes.value_counts()
        st.bar_chart(dtype_counts)
        
        # Sample data
        st.markdown("### 🔍 Sample Data")
        st.dataframe(df.head(10))
    
    elif analysis_type == "Statistical Summary":
        st.markdown("### 📈 Statistical Summary")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe())
            
            # Distribution plots
            st.markdown("### 📊 Distributions")
            selected_col = st.selectbox("Select column for distribution:", numeric_cols)
            
            if selected_col:
                st.markdown(f"#### Distribution of {selected_col}")
                
                # Create histogram
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(df[selected_col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                ax.set_xlabel(selected_col)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {selected_col}')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
        else:
            st.warning("No numeric columns found for statistical analysis!")
    
    elif analysis_type == "Column Analysis":
        st.markdown("### 🔍 Column Analysis")
        
        selected_col = st.selectbox("Select column to analyze:", df.columns)
        
        if selected_col:
            col_data = df[selected_col]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Data Type", str(col_data.dtype))
                st.metric("Non-Null Values", col_data.count())
            
            with col2:
                if pd.api.types.is_numeric_dtype(col_data):
                    st.metric("Mean", f"{col_data.mean():.2f}")
                    st.metric("Std Dev", f"{col_data.std():.2f}")
                else:
                    st.metric("Unique Values", col_data.nunique())
                    st.metric("Most Common", col_data.mode().iloc[0] if not col_data.mode().empty else "N/A")
            
            with col3:
                if pd.api.types.is_numeric_dtype(col_data):
                    st.metric("Min", f"{col_data.min():.2f}")
                    st.metric("Max", f"{col_data.max():.2f}")
                else:
                    st.metric("Empty Values", col_data.isnull().sum())
                    st.metric("Empty %", f"{col_data.isnull().sum() / len(col_data) * 100:.1f}%")
            
            # Show sample values
            st.markdown("### 📋 Sample Values")
            if pd.api.types.is_numeric_dtype(col_data):
                st.write(col_data.describe())
            else:
                value_counts = col_data.value_counts().head(10)
                st.bar_chart(value_counts)
    
    elif analysis_type == "Missing Values Analysis":
        st.markdown("### ❓ Missing Values Analysis")
        
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing %': missing_percent
        }).sort_values('Missing Count', ascending=False)
        
        # Only show columns with missing values
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if len(missing_df) > 0:
            st.dataframe(missing_df)
            
            # Visualization
            st.markdown("### 📊 Missing Values Visualization")
            st.bar_chart(missing_df['Missing %'])
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            high_missing = missing_df[missing_df['Missing %'] > 50]
            if len(high_missing) > 0:
                st.warning(f"Columns with >50% missing values: {', '.join(high_missing.index)}")
                st.info("Consider dropping these columns or imputing with domain knowledge.")
            
            medium_missing = missing_df[(missing_df['Missing %'] > 10) & (missing_df['Missing %'] <= 50)]
            if len(medium_missing) > 0:
                st.info(f"Columns with 10-50% missing values: {', '.join(medium_missing.index)}")
                st.info("Consider mean/median imputation for numeric columns or mode for categorical.")
            
            low_missing = missing_df[missing_df['Missing %'] <= 10]
            if len(low_missing) > 0:
                st.success(f"Columns with <10% missing values: {', '.join(low_missing.index)}")
                st.success("These columns have good data quality.")
        else:
            st.success("🎉 No missing values found in the dataset!")

# About page
elif page == "ℹ️ About":
    st.markdown("## ℹ️ About ProjekData")
    
    st.markdown("""
    ### 🚀 ProjekData - AI Data Analysis Platform
    
    **Version:** Simple Demo v1.0
    
    **Description:**
    ProjekData is a comprehensive AI-powered data analysis platform designed to make data science accessible to everyone.
    
    ### ✨ Features (Simple Version)
    
    - 📁 **Data Upload**: Upload CSV files or use sample datasets
    - 📊 **Data Analysis**: Statistical analysis and data exploration
    - 🔍 **Data Quality**: Missing values analysis and data profiling
    - 📈 **Visualizations**: Basic charts and distributions
    
    ### 🛠️ Technologies Used
    
    - **Streamlit**: Web application framework
    - **Pandas**: Data manipulation and analysis
    - **NumPy**: Numerical computing
    - **Matplotlib**: Data visualization
    
    ### 📚 Sample Datasets
    
    1. **Sales Data**: Retail sales transactions
    2. **Customer Reviews**: Customer feedback with sentiment
    3. **Iris Dataset**: Classic classification dataset
    4. **Indonesian News**: News articles in Bahasa Indonesia
    5. **Financial Data**: Stock market time series data
    
    ### 🎯 How to Use
    
    1. **Upload Data**: Go to Data Upload page and upload your CSV file
    2. **Explore**: Use Data Analysis page to explore your data
    3. **Analyze**: Get insights through statistical analysis
    4. **Visualize**: Create charts to understand patterns
    
    ### 🚀 Full Version Features
    
    The full version includes:
    - 🤖 AI Assistant with Gemini 2.5 Flash
    - 🤖 Machine Learning with 15+ algorithms
    - 📊 Advanced visualizations with Plotly
    - 📝 Text analytics and NLP
    - 🤝 Real-time collaboration
    - 📤 Export capabilities
    - 🔧 Deployment options
    
    ### 📞 Contact & Support
    
    - **Documentation**: Check the `docs/` folder
    - **Quick Start**: See `QUICK_START.md`
    - **Troubleshooting**: See `CARA_JALANKAN.md`
    
    ---
    
    **Made with ❤️ in Indonesia**
    
    *© 2024 ProjekData Team*
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "🚀 ProjekData - AI Data Analysis Platform | Simple Demo Version"
    "</div>",
    unsafe_allow_html=True
)