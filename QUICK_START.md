# 🚀 Quick Start Guide - ProjekData

Guide cepat untuk memulai AI Data Analysis Platform dalam 5 menit.

## 📋 Prerequisites

- Python 3.9+ terinstall
- Git (opsional)
- Gemini API Key (dapatkan dari [Google AI Studio](https://makersuite.google.com/app/apikey))

## ⚡ Quick Installation

### Option 1: Automated Setup (Recommended)

```bash
# Clone atau download project
cd projekdata

# Run setup script
python setup.py
```

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download spaCy model
python -m spacy download en_core_web_sm

# Create necessary directories
mkdir uploads exports logs models notebooks temp
```

## 🔑 API Key Setup

1. Dapatkan Gemini API Key dari [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Edit file `.env`:
```bash
GEMINI_API_KEY=your_actual_api_key_here
```

## 🏃‍♂️ Run Application

```bash
# Start Streamlit app
streamlit run app.py

# Atau dengan custom port
streamlit run app.py --server.port 8501
```

Aplikasi akan otomatis terbuka di browser:
- Local: http://localhost:8501
- Network: http://your-ip:8501

## 📊 Try Sample Data

1. Buka aplikasi di browser
2. Klik "📊 Load Sample Data" di halaman Home
3. Pilih salah satu sample datasets:
   - Sales Data (penjualan retail)
   - Customer Reviews (review pelanggan)
   - Iris Dataset (klasifikasi bunga)
   - Indonesian News (berita Bahasa Indonesia)
   - Financial Data (data saham)

## 🎯 Quick Features to Try

### 1. Data Analysis
- Upload file CSV/Excel
- Lihat data overview dan statistics
- Jalankan data quality assessment

### 2. Machine Learning
- Pilih target variable
- AutoML akan memilih model terbaik
- Lihat feature importance dan model performance

### 3. Visualizations
- Buat interactive charts
- Scatter plots, histograms, heatmaps
- Customizable colors dan themes

### 4. AI Assistant
- Chat dengan AI untuk analisis data
- Mintai rekomendasi visualisasi
- Dapatkan insights otomatis

### 5. Text Analytics
- Sentiment analysis
- Word cloud generation
- Topic modeling

## 🔧 Common Issues & Solutions

### Issue: "streamlit not found"
```bash
pip install streamlit
```

### Issue: "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### Issue: "NLTK data not found"
```bash
python -c "import nltk; nltk.download('punkt')"
```

### Issue: "spaCy model not found"
```bash
python -m spacy download en_core_web_sm
```

### Issue: "Gemini API key invalid"
- Pastikan API key benar dan aktif
- Check quota di Google AI Studio
- Restart aplikasi setelah update .env

## 📱 Mobile Access

Aplikasi juga bisa diakses dari mobile device:
1. Pastikan device dan computer di network yang sama
2. Buka http://computer-ip:8501 di mobile browser
3. Enable "Network access" saat start aplikasi

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Docker
```bash
docker build -t projekdata .
docker run -p 8501:8501 projekdata
```

### Cloud (Heroku/Railway)
1. Push ke GitHub
2. Connect ke deployment platform
3. Set environment variables
4. Deploy!

## 📚 Next Steps

1. **Explore Documentation**: [docs/](docs/)
2. **Try Advanced Features**: ML models, text analytics
3. **Customize**: Modify themes, add custom models
4. **Deploy**: Share dengan team atau public

## 🆘 Need Help?

- 📖 [Full Documentation](docs/README.md)
- 🐛 [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
- 🚀 [Deployment Guide](docs/DEPLOYMENT.md)
- 💬 [Community Support](https://discord.gg/projekdata)

## 🎉 Success!

Jika semua berjalan lancar, kamu sekarang memiliki:
- ✅ AI Data Analysis Platform yang running
- ✅ Sample datasets untuk eksplorasi
- ✅ ML models untuk prediksi
- ✅ Interactive visualizations
- ✅ AI assistant untuk bantuan

**Selamat menganalisis data! 🎊**

---

*Quick Start Guide v1.0 | ProjekData Team*