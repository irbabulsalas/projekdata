# 🚀 AI Data Analysis Platform - ProjekData

Platform analisis data berbasis AI yang komprehensif dengan integrasi Gemini 2.5 Flash, Machine Learning otomatis, dan visualisasi interaktif.

## ✨ Fitur Utama

### 🤖 AI Chat Assistant
- **Gemini 2.5 Flash Integration**: Asisten AI cerdas dengan kemampuan function calling
- **Multi-language Support**: Dukungan Bahasa Indonesia dan Inggris
- **Contextual Analysis**: Analisis data berdasarkan konteks percakapan
- **Smart Recommendations**: Rekomendasi analisis otomatis

### 📊 Machine Learning Otomatis
- **15+ Algoritma ML**: Random Forest, XGBoost, LightGBM, CatBoost, Neural Networks
- **AutoML Pipeline**: Pemilihan model otomatis dengan hyperparameter tuning
- **Ensemble Methods**: Kombinasi multiple model untuk akurasi maksimal
- **Model Interpretability**: SHAP values untuk explainable AI

### 📈 Visualisasi Interaktif
- **Multi-page Dashboard**: Interface responsif dengan navigasi intuitif
- **Real-time Charts**: Grafik interaktif dengan Plotly
- **Custom Themes**: Tema visualisasi yang dapat disesuaikan
- **Export Capabilities**: Export visualisasi ke berbagai format

### 🔍 Text Analytics
- **Sentiment Analysis**: Analisis sentimen dengan TextBlob dan VADER
- **Topic Modeling**: LDA dan NMF untuk topic discovery
- **Word Clouds**: Visualisasi frekuensi kata yang menarik
- **Indonesian NLP**: Dukungan khusus Bahasa Indonesia dengan Sastrawi

### 📁 Smart Data Pipeline
- **Auto-detection**: Deteksi otomatis tipe data dan pola
- **Quality Assessment**: Penilaian kualitas data otomatis
- **Adaptive Processing**: Pemrosesan yang menyesuaikan dengan karakteristik data
- **Error Recovery**: Penanganan error yang robust

### 🤝 Kolaborasi Real-time
- **Multi-user Support**: Dukungan pengguna simultan
- **Version Control**: Tracking perubahan analisis
- **Live Sharing**: Berbagi hasil analisis real-time
- **Security Features**: Enkripsi dan akses kontrol

### 📤 Export Komprehensif
- **Multiple Formats**: CSV, Excel, JSON, PDF
- **Model Export**: Export model terlatih (pickle, joblib)
- **Jupyter Notebooks**: Generate notebook otomatis
- **Report Generation**: Laporan analisis otomatis

## 🛠️ Teknologi

### Backend
- **Python 3.9+**: Bahasa pemrograman utama
- **Streamlit**: Web framework untuk aplikasi data
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation dan analysis
- **numpy**: Numerical computing

### AI & ML
- **Google Gemini 2.5 Flash**: AI model integration
- **XGBoost/LightGBM/CatBoost**: Advanced ML algorithms
- **SHAP**: Model interpretability
- **NLTK/spaCy**: Natural language processing
- **Sastrawi**: Indonesian NLP library

### Visualization
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Statistical plotting
- **WordCloud**: Text visualization

### Deployment
- **Docker**: Containerization
- **Redis**: Caching dan session management
- **WebSockets**: Real-time communication

## 📦 Installation

### Prerequisites
```bash
Python 3.9+
pip atau conda
```

### Setup
1. Clone repository:
```bash
git clone <repository-url>
cd projekdata
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Setup environment variables:
```bash
cp .env.example .env
# Edit .env dan tambahkan Gemini API key
```

4. Download NLTK data:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

5. Run aplikasi:
```bash
streamlit run app.py
```

## 🚀 Quick Start

### 1. Upload Data
- Drag & drop file CSV/Excel
- Atau gunakan sample datasets yang tersedia

### 2. AI Assistant
- Chat dengan AI assistant untuk analisis
- Mintai rekomendasi visualisasi
- Dapatkan insight otomatis

### 3. Machine Learning
- Pilih target variable
- AutoML akan memilih model terbaik
- Interpret hasil dengan SHAP

### 4. Visualisasi
- Buat dashboard interaktif
- Customisasi tampilan
- Export hasil

### 5. Export & Share
- Export analisis ke PDF
- Bagikan notebook
- Kolaborasi dengan tim

## 📁 Project Structure

```
projekdata/
├── modules/                 # Core modules
│   ├── smart_pipeline.py   # Smart data analysis pipeline
│   ├── data_processing.py  # Advanced data processing
│   ├── ml_models.py        # Machine learning models
│   ├── visualizations.py   # Interactive visualizations
│   ├── text_analytics.py   # Text analytics & NLP
│   ├── gemini_integration.py # AI assistant integration
│   ├── export_handler.py   # Export functionality
│   └── collaboration.py    # Real-time collaboration
├── utils/                  # Utility modules
│   ├── error_handler.py    # Error handling
│   ├── rate_limiter.py     # API rate limiting
│   └── helpers.py          # Helper functions
├── assets/                 # Static assets
│   └── sample_datasets/    # Sample datasets
├── docs/                   # Documentation
├── requirements.txt        # Dependencies
├── .env                    # Environment variables
└── app.py                  # Main application
```

## 🎯 Use Cases

### Business Intelligence
- Sales analysis dan forecasting
- Customer behavior analysis
- Market trend identification

### Data Science
- Exploratory data analysis
- Model development
- Feature engineering

### Academic Research
- Statistical analysis
- Text mining
- Visualization for publications

### Financial Analysis
- Time series analysis
- Risk assessment
- Portfolio optimization

## 🔧 Configuration

### Environment Variables
```bash
GEMINI_API_KEY=your_gemini_api_key_here
REDIS_URL=redis://localhost:6379
DEBUG=False
```

### Customization
- Edit `config.py` untuk pengaturan aplikasi
- Modifikasi themes di `assets/themes/`
- Tambah custom models di `modules/custom_models/`

## 📚 Sample Datasets

Platform menyediakan sample datasets untuk testing:

1. **sample_sales.csv** - Data penjualan retail
2. **customer_reviews.csv** - Review pelanggan dengan sentimen
3. **iris_dataset.csv** - Classic iris classification dataset
4. **indonesian_news.csv** - Berita Bahasa Indonesia untuk NLP
5. **financial_data.csv** - Data saham untuk time series analysis

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

Project ini dilisensikan under MIT License - lihat [LICENSE](LICENSE) file untuk details.

## 🆘 Support

### Documentation
- [User Guide](docs/user_guide.md)
- [API Reference](docs/api_reference.md)
- [Troubleshooting](docs/troubleshooting.md)

### Community
- [Discord Server](https://discord.gg/projekdata)
- [GitHub Issues](https://github.com/projekdata/issues)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/projekdata)

## 🔄 Updates & Changelog

### Version 1.0.0
- Initial release
- Core ML functionality
- Gemini integration
- Basic visualizations

### Upcoming Features
- Advanced time series analysis
- Image processing capabilities
- Enhanced collaboration features
- Mobile app support

## 🌟 Acknowledgments

- Google Gemini Team untuk API yang amazing
- Streamlit community untuk inspiration
- Open source contributors
- Beta testers dan early adopters

---

**Made with ❤️ by ProjekData Team**

*Empowering data-driven decisions with AI*