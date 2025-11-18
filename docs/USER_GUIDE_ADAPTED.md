# 📖 User Guide - AI Data Analysis Platform (Adapted Version)

## Selamat Datang!

Platform Analisis Data AI yang telah diadaptasi dengan gaya dan preferensi proyek referensi fungsional-caling. Platform ini dikembangkan oleh **Muhammad Irbabul Salas** dengan fitur-fitur canggih untuk analisis data otomatis.

## 🚀 Memulai Platform

### 1. Login & Authentication
- Buka aplikasi di http://localhost:8503
- Login atau register melalui sidebar
- Masukkan username, email, dan password
- Klik "Login" untuk masuk ke platform

### 2. Upload Data
- Di sidebar, klik "Upload your dataset"
- Pilih file (CSV, Excel, JSON, Parquet)
- Atau gunakan sample datasets:
  - **📝 Load Sample E-commerce Data** - Data penjualan e-commerce
  - **💬 Load Sample Reviews Data** - Data ulasan pelanggan

### 3. Navigasi Platform
Gunakan radio button di sidebar untuk berpindah antar halaman:
- **📈 Overview** - Dashboard utama dengan metrik data
- **🔍 Data Profiling** - Analisis mendalam tentang data
- **📊 EDA** - Exploratory Data Analysis
- **🤖 ML Models** - Machine Learning Models
- **📝 Text Analytics** - Analisis teks
- **💾 Projects** - Simpan & load projects
- **📥 Export Center** - Export hasil analisis

## 📈 Fitur Utama

### Overview Dashboard
- **Metrik Data**: Total rows, columns, missing values, quality score
- **Tipe Kolom**: Visualisasi distribusi tipe data
- **Preview Data**: Tampilan 10 baris pertama
- **AI Insights**: Generate insights otomatis dengan Gemini AI

### Data Profiling
- **Informasi Dasar**: Shape, duplicates, quality score
- **Detail Kolom**: Tipe data, missing values, unique values
- **Correlation Heatmap**: Korelasi antar variabel numerik
- **Data Cleaning**: 
  - Handle missing values (mean, median, mode, drop)
  - Remove duplicates
  - Handle outliers (IQR, Z-score)

### Exploratory Data Analysis (EDA)
#### 📈 Distributions
- Histogram untuk distribusi data
- Box plot untuk identifikasi outliers
- Pilih kolom numerik untuk analisis

#### 🎯 Relationships
- Scatter plot untuk hubungan antar variabel
- Color coding berdasarkan kategori
- Pilih X dan Y axis secara dinamis

#### 📦 Comparisons
- Violin plot untuk perbandingan distribusi
- Group by categorical variables
- Visualisasi perbandingan antar grup

### Machine Learning Models
#### 🎯 Classification
- Pilih target column untuk klasifikasi
- Pilih model algorithms:
  - Random Forest
  - XGBoost
  - Logistic Regression
  - LightGBM
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix**: Visualisasi performa
- **Feature Importance**: Pentingnya setiap fitur

#### 📈 Regression
- Pilih target column numerik
- Model algorithms:
  - Random Forest
  - XGBoost
  - Ridge Regression
  - Lasso Regression
- **Metrics**: RMSE, MAE, R² Score
- **Best Model**: Model dengan performa terbaik

#### 🔍 Clustering
- Pilih jumlah cluster (2-10)
- Pilih metode clustering:
  - K-Means
  - DBSCAN
- Visualisasi cluster dalam 2D/3D

### Text Analytics
- **Pilih Kolom Teks**: Pilih kolom dengan data teks
- **Analisis Sentimen**: Deteksi emosi dalam teks
- **Word Cloud**: Visualisasi kata yang sering muncul
- **Bigrams**: Pasangan kata yang sering muncul
- **Statistik Teks**: Jumlah kata, panjang rata-rata

### Projects Management
#### 💾 Save Project
- Masukkan nama project
- Tambahkan deskripsi (opsional)
- Pilih public/private
- Otomatis menyimpan data dan model

#### 📂 Load Project
- Daftar semua project yang tersimpan
- Tampilkan informasi project
- Load kembali project yang dipilih

#### 📊 My Datasets
- Daftar semua dataset yang tersimpan
- Informasi ukuran dan kolom
- Load dataset kembali ke analisis

#### 🤖 My Models
- Daftar model yang telah dilatih
- Informasi algoritma dan metrik
- Timestamp pembuatan model

### Export Center
- **Data Export**: CSV, Excel, JSON, Parquet
- **Model Export**: .pkl, .joblib format
- **PDF Report**: Laporan analisis lengkap
- **Jupyter Notebook**: Export ke .ipynb

## 💬 AI Chat Assistant

### Cara Menggunakan
1. Ketik pertanyaan tentang data di chat box
2. Klik "📤 Send" untuk mengirim
3. AI akan merespon dengan insight atau analisis

### Contoh Pertanyaan
- "Tampilkan korelasi antara usia dan gaji"
- "Train model klasifikasi untuk prediksi churn"
- "Analisis sentimen dari customer reviews"
- "Buat visualisasi untuk distribusi harga"
- "Identifikasi outliers dalam data penjualan"

### Rate Limiting
- **1 menit** antara pertanyaan
- **15 pertanyaan per jam** (free tier)
- Status rate limit ditampilkan di atas chat

## 🎨 Personalisasi

### Header Profile
- Nama: Muhammad Irbabul Salas
- Foto profil (opsional)
- Theme toggle: 🌙/☀️

### Responsive Design
- **Mobile**: Layout otomatis menyesuaikan
- **Tablet**: Sidebar compact
- **Desktop**: Full layout dengan semua fitur

## 🔧 Tips & Tricks

### Data Upload
- Gunakan CSV untuk performa terbaik
- Pastikan header jelas dan konsisten
- Hapus data yang tidak perlu sebelum upload

### Machine Learning
- Minimum 50 rows untuk training
- Pilih target column yang relevan
- Coba multiple algorithms untuk perbandingan
- Periksa feature importance untuk insight

### Text Analytics
- Pastikan kolom teks dalam Bahasa Indonesia/Inggris
- Hapus special characters jika perlu
- Gunakan sample data untuk testing

### Performance
- Gunakan sample datasets untuk mencoba fitur
- Clear chat history untuk performa lebih baik
- Save projects secara berkala

## 🚨 Troubleshooting

### Common Issues

**Q: "GEMINI_API_KEY not found"**
A: Tambahkan API key ke file .env:
```
GEMINI_API_KEY=your-api-key-here
```

**Q: "File upload failed"**
A: Periksa format file (CSV, Excel, JSON, Parquet) dan ukuran maksimal 200MB

**Q: "Model training failed"**
A: Pastikan:
- Minimum 50 rows data
- Target column terpilih
- Tidak ada missing values di target

**Q: "Rate limit exceeded"**
A: Tunggu 1 menit antara pertanyaan, maksimal 15/jam

**Q: "Database features unavailable"**
A: Install dependencies:
```bash
pip install bcrypt sqlite3
```

### Error Messages
- **❌ Error**: Kesalahan fatal, perlu diperbaiki
- **⚠️ Warning**: Peringatan, bisa diabaikan
- **ℹ️ Info**: Informasi penting
- **✅ Success**: Operasi berhasil

## 📞 Support

### Help & Support
- **Quick Guide**: Di sidebar bawah
- **Keyboard Shortcuts**: Ctrl+U (upload), Ctrl+/ (help)
- **Documentation**: Lihat folder docs/

### Contact
- Developer: Muhammad Irbabul Salas
- Platform: AI Data Analysis Platform
- Powered by: Google Gemini 2.5 Flash

## 🔄 Updates & Features

### Current Version: 2.0.0
- ✅ Authentication & database
- ✅ Personal header & profile
- ✅ Enhanced UI/UX
- ✅ Project management
- ✅ Rate limiting
- ✅ Responsive design

### Coming Soon
- 🔄 Real-time collaboration
- 📊 Advanced visualizations
- 🤖 More ML algorithms
- 🌐 Multi-language support
- 📱 Mobile app

---

## 🎉 Selamat Menggunakan!

Terima kasih telah menggunakan AI Data Analysis Platform versi adaptasi. Platform ini dirancang untuk membuat analisis data menjadi mudah, cepat, dan menyenangkan dengan bantuan AI.

**Created with ❤️ by Muhammad Irbabul Salas**

*Powered by Google Gemini 2.5 Flash | Built with Streamlit*