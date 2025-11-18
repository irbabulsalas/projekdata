# 🚀 Cara Menjalankan ProjekData

Panduan lengkap step-by-step untuk menjalankan AI Data Analysis Platform.

## 📋 Prerequisites

Pastikan kamu sudah memiliki:
- **Python 3.9+** (cek dengan `python --version`)
- **Gemini API Key** (dapatkan dari [Google AI Studio](https://makersuite.google.com/app/apikey))

## 🔧 Cara 1: Otomatis (Recommended)

### Step 1: Buka Terminal/CMD
- **Windows**: Tekan `Win + R`, ketik `cmd`, tekan Enter
- **Mac**: Buka Terminal dari Applications
- **Linux**: Buka Terminal (Ctrl+Alt+T)

### Step 2: Navigasi ke Folder ProjekData
```bash
cd C:\Users\LEGION\Documents\projekdata
```

### Step 3: Jalankan Setup Otomatis
```bash
python setup.py
```

Setup akan:
- ✅ Install semua dependencies
- ✅ Download NLTK data
- ✅ Download spaCy model
- ✅ Buat folder yang diperlukan
- ✅ Test semua imports

### Step 4: Setup API Key
Edit file `.env` dan ganti dengan API key kamu:
```bash
GEMINI_API_KEY=masukkan_api_key_kamu_disini
```

### Step 5: Jalankan Aplikasi
```bash
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser: http://localhost:8501

---

## 🔧 Cara 2: Manual (Jika Cara 1 Gagal)

### Step 1: Install Dependencies
```bash
pip install streamlit pandas numpy plotly scikit-learn
pip install google-generativeai nltk spacy textblob
pip install xgboost lightgbm catboost shap
pip install wordcloud matplotlib seaborn fpdf
pip install redis websockets openpyxl
```

### Step 2: Install Requirements Lengkap
```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('stopwords')"
python -c "import nltk; nltk.download('wordnet')"
python -c "import nltk; nltk.download('vader_lexicon')"
```

### Step 4: Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### Step 5: Buat Folder yang Diperlukan
```bash
mkdir uploads exports logs models notebooks temp
```

### Step 6: Setup API Key
Buka file `.env` dan edit:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 7: Jalankan Aplikasi
```bash
streamlit run app.py
```

---

## 🐛 Troubleshooting

### Problem: "streamlit not found"
**Solution:**
```bash
pip install streamlit
```

### Problem: "ModuleNotFoundError"
**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: "NLTK data not found"
**Solution:**
```bash
python -c "import nltk; nltk.download('punkt')"
```

### Problem: "spaCy model not found"
**Solution:**
```bash
python -m spacy download en_core_web_sm
```

### Problem: "Permission denied"
**Solution:**
- **Windows**: Run CMD as Administrator
- **Mac/Linux**: Gunakan `sudo` atau cek permissions

### Problem: "Port 8501 already in use"
**Solution:**
```bash
streamlit run app.py --server.port 8502
```

### Problem: "API key invalid"
**Solution:**
1. Pastikan API key benar dari Google AI Studio
2. Check quota di Google AI Console
3. Restart aplikasi setelah update .env

---

## 📱 Cara Akses

### Local Access
Buka browser dan pergi ke: http://localhost:8501

### Network Access (untuk akses dari device lain)
1. Saat menjalankan, gunakan:
```bash
streamlit run app.py --server.address 0.0.0.0
```
2. Akses dari device lain: http://ip_komputer:8501

### Mobile Access
Aplikasi responsive dan bisa diakses dari mobile browser dengan URL yang sama.

---

## 🎯 Quick Test Setelah Berjalan

1. **Buka aplikasi** di browser
2. **Klik "📊 Load Sample Data"** di halaman Home
3. **Pilih dataset** (misal: Sales Data)
4. **Coba fitur:**
   - Data Analysis → Statistical Summary
   - Visualizations → Scatter Plot
   - Machine Learning → Classification
   - AI Assistant → Chat dengan AI

---

## 🔄 Cara Stop Aplikasi

### Di Terminal/CMD:
- Tekan `Ctrl + C`
- Atau tutup terminal window

### Cara Restart:
```bash
# Stop dengan Ctrl+C
# Lalu jalankan lagi:
streamlit run app.py
```

---

## 📁 Folder Structure Setelah Berjalan

```
projekdata/
├── uploads/        # File yang diupload user
├── exports/        # Hasil export
├── logs/          # Log files
├── models/        # Model yang disimpan
├── notebooks/     # Jupyter notebooks
└── temp/          # Temporary files
```

---

## 🎉 Success Indicators

Jika berhasil, kamu akan melihat:
1. ✅ Terminal menampilkan:
   ```
   You can now view your Streamlit app in your browser.
   Local URL: http://localhost:8501
   Network URL: http://your-ip:8501
   ```

2. ✅ Browser terbuka otomatis dengan halaman ProjekData

3. ✅ Halaman Home menampilkan "🚀 ProjekData - AI Data Analysis Platform"

4. ✅ Sample data bisa di-load tanpa error

---

## 🆘 Butuh Bantuan?

### Quick Help:
- **Documentation**: Buka `docs/` folder
- **Troubleshooting**: Lihat `docs/TROUBLESHOOTING.md`
- **Quick Start**: Lihat `QUICK_START.md`

### Common Issues:
1. **Python version tidak compatible** → Install Python 3.9+
2. **Internet connection** → Diperlukan untuk download dependencies
3. **API key quota** → Check limit di Google AI Console
4. **Memory issues** → Restart aplikasi atau gunakan dataset lebih kecil

### Contact Support:
- 📧 Email: support@projekdata.com
- 💬 Discord: [Join Community](https://discord.gg/projekdata)

---

**Selamat menggunakan ProjekData! 🎊**

*Jika masih ada masalah, jangan ragu untuk bertanya di issues atau community.*