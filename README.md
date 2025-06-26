# 🧠 ARSignID - Augmented Reality Sign Indonesia

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

## 🛠️ Teknologi yang Digunakan

- **Python 3.8+** - Bahasa pemrograman utama
- **Flask** - Web framework untuk backend
- **OpenCV** - Pemrosesan gambar dan computer vision
- **MediaPipe** - Library untuk deteksi tangan
- **TensorFlow** - Deep learning framework untuk klasifikasi
- **NumPy** - Manipulasi array dan operasi numerik

---

