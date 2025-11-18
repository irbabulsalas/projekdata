# 🤖 AI Data Analysis Platform (Adapted Version)
### By Muhammad Irbabul Salas

Versi adaptasi dari platform analisis data AI yang disesuaikan dengan preferensi dan gaya proyek referensi fungsional-caling.

![Platform](https://img.shields.io/badge/Platform-Streamlit-red)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![AI](https://img.shields.io/badge/AI-Gemini_2.5-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Fitur Utama

### 🎯 Kemampuan Inti
- **AI Chat Assistant** - Ditenagai oleh Gemini 2.5 Flash dengan function calling
- **Automated Machine Learning** - 10+ algoritma dengan perbandingan otomatis
- **Interactive Dashboards** - Interface multi-page yang responsif
- **Text Analytics** - Analisis sentimen, topic modeling, word clouds
- **Comprehensive Export** - PDF, Excel, models, Jupyter notebooks
- **Authentication & Database** - Sistem login dan penyimpanan project

### 📊 Analisis Data
- Multi-format upload (CSV, Excel, JSON, Parquet, TSV)
- Automatic data profiling & quality assessment  
- Advanced cleaning dengan berbagai strategi
- Statistical tests & correlation analysis
- Feature importance & SHAP values

### 🎨 Pengalaman Pengguna
- Desain responsif (mobile/tablet/desktop)
- Dark/Light mode toggle
- Interactive onboarding & help system
- Sample datasets untuk testing instan
- Rate limiting untuk free API tier
- Personal header dengan nama dan foto profil

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Gemini API Key ([Dapatkan Free Key](https://aistudio.google.com/app/apikey))

### Instalasi

1. **Clone atau download project ini**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Gemini API Key**
   - Dapatkan free API key dari: https://aistudio.google.com/app/apikey
   - Tambahkan ke file `.env` dengan key: `GEMINI_API_KEY`

4. **Jalankan aplikasi yang diadaptasi**
   ```bash
   streamlit run app_adapted.py --server.port 8503
   ```

5. **Buka browser**
   ```
   http://localhost:8503
   ```

---

## 📖 Panduan Pengguna

### Upload Data
1. Klik sidebar "Upload Data"
2. Pilih file (CSV, Excel, JSON, Parquet)
3. Atau load sample datasets untuk mencoba fitur

### AI Chat Assistant
- Ajukan pertanyaan natural language tentang data Anda
- **Rate Limit**: 1 menit antara pertanyaan, 15/jam (free tier)
- **Contoh**:
  - "Tampilkan korelasi antara usia dan gaji"
  - "Train model klasifikasi untuk prediksi churn"
  - "Analisis sentimen dari customer reviews"

### Machine Learning
1. Pergi ke tab "🤖 ML Models"
2. Pilih target column
3. Pilih models untuk training
4. Klik "Train Models"
5. Lihat metrics, confusion matrix, feature importance

### Authentication & Projects
- Login/Register melalui sidebar
- Save projects dengan data dan models
- Load kembali projects yang tersimpan
- Manage datasets dan models terlatih

---

## 🏗️ Struktur Proyek

```
ai-data-analysis-adapted/
├── app_adapted.py                  # Aplikasi utama yang diadaptasi
├── app.py                          # Aplikasi original
├── requirements.txt                 # Python dependencies
├── .env                            # Environment variables
│
├── modules/                        # Core modules
│   ├── data_processing.py          # Data loading & cleaning
│   ├── ml_models.py                # ML training & evaluation
│   ├── visualizations.py           # Chart generation
│   ├── text_analytics.py           # NLP functions
│   ├── gemini_integration.py       # AI function calling
│   └── export_handler.py           # Export functionality
│
├── utils/                         # Utilities
│   ├── error_handler.py            # Error management
│   ├── rate_limiter.py             # API rate limiting
│   └── helpers.py                  # Helper functions
│
├── database/                      # Database & authentication
│   ├── auth_manager.py             # User authentication
│   ├── session_manager.py          # Project management
│   ├── db_manager.py              # Database operations
│   └── init_db.py                 # Database initialization
│
├── assets/                        # Static files
│   ├── profile_photo.jpg           # User photo (optional)
│   └── sample_datasets/           # Sample data
│
└── docs/                          # Documentation
    ├── DEPLOYMENT.md              # Railway deployment guide
    └── TROUBLESHOOTING.md         # Common issues
```

---

## 🌐 Deployment

### Local Development
```bash
streamlit run app_adapted.py
```

### Railway Deployment
1. Push code ke GitHub
2. Connect Railway ke repo Anda
3. Add `GEMINI_API_KEY` ke environment variables
4. Deploy!

**Estimated Cost**: ~$5/bulan dengan Railway Hobby plan

---

## 💰 Cost Breakdown

| Service | Free Tier | Monthly Cost |
|---------|-----------|--------------|
| Gemini API (Flash) | 15 req/min, 1.5K/day | **FREE** |
| Railway | $5 credit trial | ~$5 setelah trial |
| GitHub | Unlimited repos | **FREE** |
| **Total** | | **~$5/bulan** |

---

## 🔑 Mendapatkan API Keys

### Gemini API (Required)
1. Kunjungi: https://aistudio.google.com/app/apikey
2. Login dengan akun Google
3. Klik "Create API Key"
4. Copy dan simpan ke file `.env`

---

## 🎯 Fitur per Dashboard

### 📈 Overview Dashboard
- Total rows, columns, missing values
- Data quality score
- Column type breakdown
- AI-generated insights

### 🔍 Data Profiling
- Detailed column statistics
- Missing values analysis
- Correlation heatmap
- Data cleaning interface

### 📊 EDA (Exploratory Data Analysis)
- Distribution plots (histogram, box, violin)
- Relationship analysis (scatter, line)
- Statistical comparisons

### 🤖 ML Models
- Classification (Random Forest, XGBoost, Logistic Regression, etc.)
- Regression (Ridge, Lasso, Random Forest)
- Clustering (K-Means, DBSCAN)
- Feature importance & SHAP values

### 📝 Text Analytics
- Sentiment analysis
- Word clouds
- N-gram analysis (bigrams, trigrams)
- Text statistics

### 💾 Projects
- Save & load projects
- Manage datasets
- Store trained models
- User authentication required

### 📥 Export Center
- Data exports (CSV, Excel, JSON, Parquet)
- Model exports (.pkl, .joblib)
- PDF reports
- Jupyter notebooks

---

## ⚙️ Tech Stack

**Frontend/UI:**
- Streamlit (web framework)
- Plotly (interactive visualizations)
- Custom CSS (responsive design)

**AI/ML:**
- Google Gemini 2.5 (AI chat & function calling)
- scikit-learn (traditional ML)
- XGBoost, LightGBM (gradient boosting)
- SHAP (model interpretability)

**Data Processing:**
- pandas (data manipulation)
- NumPy (numerical computing)
- NLTK, TextBlob (NLP)

**Database & Auth:**
- SQLite (database)
- bcrypt (password hashing)
- Session management

**Export:**
- FPDF, ReportLab (PDF generation)
- Joblib (model serialization)
- NBFormat (Jupyter notebooks)

---

## 🐛 Troubleshooting

### Common Issues

**Q: "API rate limit reached"**
A: Tunggu 1 menit antara pertanyaan. Free tier allows 15 requests/hour.

**Q: "File upload failed"**
A: Check file size (max 200MB) dan format. Coba convert ke CSV.

**Q: "Model training failed"**
A: Pastikan Anda memiliki cukup data (min 50 rows) dan numeric features.

**Q: "GEMINI_API_KEY not found"**
A: Add API key ke file `.env` atau environment variables.

**Q: "Database features unavailable"**
A: Install dependencies: `pip install bcrypt sqlite3`

---

## 📝 License

MIT License - Free to use, modify, and distribute.

---

## 👨‍💻 Author

**Muhammad Irbabul Salas**

Platform untuk automated data analysis dengan AI assistance, diadaptasi dari proyek referensi fungsional-caling.

---

## 🙏 Acknowledgments

- Google Gemini AI untuk powerful LLM capabilities
- Streamlit untuk amazing web framework
- Open source ML libraries (scikit-learn, XGBoost, etc.)
- Proyek referensi fungsional-caling untuk inspirasi desain dan fitur

---

## 📊 Version

**Version 2.0.0 - Adapted Release** (November 2025)
- ✅ Adaptasi UI/UX dari proyek referensi
- ✅ Tambahkan authentication & database
- ✅ Personal header dengan nama dan foto
- ✅ Enhanced navigation dengan radio buttons
- ✅ Project management system
- ✅ Improved responsive design

---

**Made with ❤️ by Muhammad Irbabul Salas**

*Powered by Gemini 2.5 Flash | Built with Streamlit*