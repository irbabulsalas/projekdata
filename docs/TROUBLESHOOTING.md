# 🔧 Troubleshooting Guide - ProjekData

Guide lengkap untuk mengatasi masalah yang mungkin terjadi saat menggunakan AI Data Analysis Platform.

## 📋 Table of Contents

- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [API & Integration Issues](#api--integration-issues)
- [Data Processing Issues](#data-processing-issues)
- [Machine Learning Issues](#machine-learning-issues)
- [Visualization Issues](#visualization-issues)
- [Deployment Issues](#deployment-issues)
- [Browser Issues](#browser-issues)

## 🚀 Installation Issues

### Python Version Compatibility
**Problem**: `ERROR: Package requires a different Python`
```bash
# Check Python version
python --version

# Install correct version (3.9+)
# Use pyenv for version management
pyenv install 3.9.16
pyenv local 3.9.16
```

### Dependency Conflicts
**Problem**: `ERROR: pip's dependency resolver does not currently take into account all the packages`
```bash
# Solution 1: Clean install
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Solution 2: Use virtual environment
python -m venv projekdata-env
source projekdata-env/bin/activate  # Linux/Mac
# projekdata-env\Scripts\activate  # Windows
pip install -r requirements.txt
```

### NLTK Data Missing
**Problem**: `LookupError: Resource punkt not found`
```bash
# Download NLTK data
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
"
```

### spaCy Model Missing
**Problem**: `OSError: [E050] Can't find model 'en_core_web_sm'`
```bash
# Download spaCy model
python -m spacy download en_core_web_sm

# Verify installation
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('Model loaded successfully')"
```

## ⚡ Runtime Errors

### Memory Issues
**Problem**: `MemoryError: Unable to allocate array`
```python
# Solution 1: Process data in chunks
import pandas as pd

# Read in chunks
chunk_size = 10000
chunks = pd.read_csv('large_file.csv', chunksize=chunk_size)

for chunk in chunks:
    process_chunk(chunk)

# Solution 2: Reduce memory usage
df = pd.read_csv('file.csv', dtype={'column': 'category'})
```

### File Not Found
**Problem**: `FileNotFoundError: [Errno 2] No such file or directory`
```python
# Check current directory
import os
print(os.getcwd())

# Use absolute paths
import pathlib
file_path = pathlib.Path(__file__).parent / 'data' / 'file.csv'
```

### Permission Denied
**Problem**: `PermissionError: [Errno 13] Permission denied`
```bash
# Linux/Mac
chmod 755 /path/to/directory

# Windows: Run as administrator
# Or change folder permissions
```

## 🐌 Performance Issues

### Slow Data Loading
**Problem**: Large datasets taking too long to load
```python
# Solution 1: Use optimized data types
dtypes = {
    'id': 'int32',
    'name': 'category',
    'value': 'float32'
}
df = pd.read_csv('file.csv', dtype=dtypes)

# Solution 2: Use Dask for very large datasets
import dask.dataframe as dd
ddf = dd.read_csv('large_file.csv')
```

### Slow Model Training
**Problem**: ML models taking too long to train
```python
# Solution 1: Reduce dataset size for testing
df_sample = df.sample(frac=0.1, random_state=42)

# Solution 2: Use parallel processing
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=-1)  # Use all cores

# Solution 3: Use faster algorithms
# Instead of deep learning, try LightGBM or XGBoost
```

### Memory Leaks
**Problem**: Memory usage increasing over time
```python
# Solution 1: Clear variables
import gc
del large_variable
gc.collect()

# Solution 2: Use context managers
with pd.read_csv('file.csv') as df:
    process_data(df)
```

## 🔌 API & Integration Issues

### Gemini API Key Issues
**Problem**: `Invalid API key`
```python
# Check API key
import os
api_key = os.getenv('GEMINI_API_KEY')
print(f"API Key exists: {bool(api_key)}")
print(f"API Key length: {len(api_key) if api_key else 0}")

# Test API connection
import google.generativeai as genai
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Hello")
    print("API working correctly")
except Exception as e:
    print(f"API Error: {e}")
```

### Rate Limiting
**Problem**: `Rate limit exceeded`
```python
# Solution 1: Implement exponential backoff
import time
import random

def api_call_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            wait_time = 2 ** attempt + random.uniform(0, 1)
            time.sleep(wait_time)
    raise Exception("Max retries exceeded")
```

### Redis Connection Issues
**Problem**: `ConnectionError: Redis connection failed`
```python
# Test Redis connection
import redis
try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
    print("Redis connection successful")
except redis.ConnectionError as e:
    print(f"Redis connection failed: {e}")
    print("Make sure Redis server is running")
```

## 📊 Data Processing Issues

### Missing Values
**Problem**: Dataset contains too many missing values
```python
# Analyze missing values
import pandas as pd

# Check missing value percentage
missing_percent = df.isnull().sum() / len(df) * 100
print(missing_percent[missing_percent > 0])

# Handle missing values
# Option 1: Drop columns with too many missing values
threshold = 0.5  # Drop columns with >50% missing
df = df.dropna(threshold=int(len(df) * (1 - threshold)), axis=1)

# Option 2: Impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[['numeric_col']] = imputer.fit_transform(df[['numeric_col']])
```

### Data Type Issues
**Problem**: Incorrect data types causing errors
```python
# Check data types
print(df.dtypes)

# Convert data types
df['date_column'] = pd.to_datetime(df['date_column'])
df['category_column'] = df['category_column'].astype('category')
df['numeric_column'] = pd.to_numeric(df['numeric_column'], errors='coerce')
```

### Encoding Issues
**Problem**: `UnicodeDecodeError`
```python
# Try different encodings
encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

for encoding in encodings:
    try:
        df = pd.read_csv('file.csv', encoding=encoding)
        print(f"Successfully read with {encoding}")
        break
    except UnicodeDecodeError:
        continue
```

## 🤖 Machine Learning Issues

### Model Not Converging
**Problem**: Model training not converging
```python
# Solution 1: Adjust hyperparameters
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=100,  # Increase trees
    max_depth=10,     # Limit depth
    random_state=42
)

# Solution 2: Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Solution 3: Feature selection
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
```

### Overfitting
**Problem**: Model performs well on training data but poorly on test data
```python
# Solution 1: Cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(f"CV Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Solution 2: Regularization
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=0.1, penalty='l2')  # Stronger regularization

# Solution 3: More training data
# Collect more data or use data augmentation
```

### Imbalanced Dataset
**Problem**: Model biased towards majority class
```python
# Solution 1: Class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
model.fit(X, y, sample_weight=class_weights[y])

# Solution 2: Resampling
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Solution 3: Different metrics
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
```

## 📈 Visualization Issues

### Plotly Charts Not Displaying
**Problem**: Charts not showing in Streamlit
```python
# Solution 1: Use st.plotly_chart
import plotly.express as px
fig = px.scatter(df, x='x', y='y')
st.plotly_chart(fig, use_container_width=True)

# Solution 2: Check plotly version
import plotly
print(f"Plotly version: {plotly.__version__}")

# Solution 3: Update plotly if needed
pip install --upgrade plotly
```

### Performance Issues with Large Datasets
**Problem**: Visualizations are slow with large data
```python
# Solution 1: Sample data
df_sample = df.sample(n=10000, random_state=42)

# Solution 2: Use appropriate chart types
# For large datasets, use:
# - 2D histograms instead of scatter plots
# - Box plots instead of violin plots
# - Aggregated bar charts

# Solution 3: Data aggregation
df_agg = df.groupby('category').agg({
    'value': ['mean', 'count']
}).reset_index()
```

### Color Scheme Issues
**Problem**: Colors not displaying correctly
```python
# Solution 1: Use colorblind-friendly palettes
import plotly.express as px
fig = px.scatter(df, x='x', y='y', color='category',
                 color_discrete_sequence=px.colors.qualitative.Set3)

# Solution 2: Custom color mapping
color_map = {
    'category1': '#FF6B6B',
    'category2': '#4ECDC4',
    'category3': '#45B7D1'
}
```

## 🚀 Deployment Issues

### Heroku Deployment Failed
**Problem**: Build failed on Heroku
```bash
# Check build logs
heroku logs --tail --app your-app-name

# Common solutions:
# 1. Check Python version in runtime.txt
echo "python-3.9.16" > runtime.txt

# 2. Verify requirements.txt format
pip freeze > requirements.txt

# 3. Check Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile
```

### Railway Deployment Issues
**Problem**: Application not starting
```bash
# Check railway.json configuration
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"
  }
}

# Check environment variables
railway variables list
```

### Docker Issues
**Problem**: Container fails to start
```bash
# Check Docker logs
docker logs container_name

# Debug container interactively
docker run -it --entrypoint /bin/bash projekdata

# Common fixes:
# 1. Check file permissions
# 2. Verify environment variables
# 3. Ensure all dependencies are installed
```

## 🌐 Browser Issues

### WebSocket Connection Failed
**Problem**: Real-time features not working
```python
# Check WebSocket configuration
# In streamlit config.toml:
[server]
enableCORS = false
enableXsrfProtection = false

# Check browser console for errors
# F12 -> Console tab
```

### File Upload Issues
**Problem**: File upload not working
```python
# Check file size limits
# In streamlit config.toml:
[server]
maxUploadSize = 200  # MB

# Check file type restrictions
import streamlit as st
uploaded_file = st.file_uploader(
    "Choose a file",
    type=['csv', 'xlsx', 'json'],
    accept_multiple_files=False
)
```

### Mobile Responsiveness
**Problem**: App not displaying correctly on mobile
```python
# Use responsive design
import streamlit as st

# Use columns for layout
col1, col2 = st.columns([1, 1])
with col1:
    st.metric("Metric 1", value1)
with col2:
    st.metric("Metric 2", value2)

# Use container width for charts
st.plotly_chart(fig, use_container_width=True)
```

## 📞 Getting Help

### Debug Mode
```python
# Enable debug mode
import streamlit as st
st.set_option('logger.level', 'debug')

# Or in .streamlit/config.toml:
[logger]
level = "debug"
```

### Error Reporting
```python
# Custom error handling
try:
    risky_operation()
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)  # Shows full traceback
```

### Community Support
- GitHub Issues: [Create an issue](https://github.com/projekdata/issues)
- Discord: [Join our community](https://discord.gg/projekdata)
- Documentation: [Full docs](https://docs.projekdata.com)

### Contact Support
- Email: support@projekdata.com
- Response time: 24-48 hours
- Include: Error message, steps to reproduce, environment details

---

**Need more help? Check our [User Guide](user_guide.md) or [API Reference](api_reference.md)**

*Happy analyzing! 🚀*