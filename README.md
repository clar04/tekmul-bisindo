# 🧠 ARSignID - Augmented Reality Sign Indonesia
![image](https://github.com/user-attachments/assets/88dbc136-88e0-4116-be46-a9e198be8bff)

Proyek ini bertujuan untuk mendeteksi bahasa isyarat Indonesia (BISINDO) menggunakan webcam secara real-time dan model machine learning yang telah dilatih. Proyek ini dibangun dengan Python, Flask sebagai backend server, OpenCV untuk pemrosesan gambar, MediaPipe untuk deteksi tangan, dan TensorFlow untuk klasifikasi isyarat.

## 👥 Tim Pengembang

**Kelompok 3 - Teknologi Multimedia A**

- **Jeany Aurellia** - 5027221008
- **Clara Valentina** - 5027221028  
- **Monika Damelia H** - 5027221011
- **Atha Rahma A** - 5027221030

---

## 📦 Prasyarat Instalasi

Pastikan Anda telah menginstal:

- **Python 3.8+**  
- **pip** (Python package manager)

Install semua dependensi dengan perintah berikut:

```bash
pip install flask opencv-python mediapipe tensorflow numpy
```

---

## 🚀 Cara Menjalankan Proyek

### 1. Clone repository

```bash
git clone https://github.com/clar04/tekmul-bisindo.git
```

### 2. Masuk ke folder proyek

```bash
cd tekmul-bisindo
```

### 3. Pastikan file model tersedia

Letakkan file model deep learning `sign_model.h5` ke dalam folder:

```
./model/sign_model.h5
```

### 4. Jalankan aplikasi

```bash
python app.py
```

### 5. Akses aplikasi di browser

Buka:

```
http://localhost:5000
```

### 6. Mulai deteksi BISINDO

Gunakan antarmuka web untuk  menggunakan webcam 

---

## ✅ Fitur

- Deteksi tangan menggunakan **MediaPipe**
- Klasifikasi huruf A–Z menggunakan model **TensorFlow**
- Antarmuka web berbasis HTML

---

## ❗ Troubleshooting

### Model not loaded
Pastikan file `sign_model.h5` ada di dalam folder `model/` dan sesuai dengan format `(50, 50)` input image.

### MediaPipe Warnings
Sudah disuppress otomatis menggunakan:

```python
os.environ['GLOG_minloglevel'] = '2'
```

---

## 📊 Dataset
Proyek ini menggunakan dataset BISINDO (Bahasa Isyarat Indonesia) yang tersedia di Kaggle:
[Dataset: Indonesian Sign Language (BISINDO)](https://www.kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo)
Dataset ini berisi koleksi gambar bahasa isyarat Indonesia untuk huruf A-Z yang digunakan untuk melatih model machine learning dalam mengenali gesture tangan.

---

## 🛠️ Teknologi yang Digunakan

- **Python 3.8+** - Bahasa pemrograman utama
- **Flask** - Web framework untuk backend
- **OpenCV** - Pemrosesan gambar dan computer vision
- **MediaPipe** - Library untuk deteksi tangan
- **TensorFlow** - Deep learning framework untuk klasifikasi
- **NumPy** - Manipulasi array dan operasi numerik

---

