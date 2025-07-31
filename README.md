
# Indonesian License Plate OCR with VLM (LLaVA) and CER Evaluation

Proyek ini bertujuan untuk melakukan Optical Character Recognition (OCR) pada plat nomor kendaraan Indonesia menggunakan model Visual Language Model (VLM) seperti **LLaVA (llava-llama-3-8b-v1_1)** yang dijalankan melalui **LM Studio**. Evaluasi dilakukan menggunakan metrik **Character Error Rate (CER)**.

---

## Dataset

Dataset yang digunakan adalah **[Indonesian License Plate Dataset](https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset)** dari Kaggle. Dataset terdiri dari:

- Folder `images/test/` → berisi gambar plat nomor
- Folder `labels/test/` → berisi label ground truth dalam format YOLO
- File `classes.names` → daftar karakter kelas (huruf & angka)

---

## Tahapan Proyek

### 1. Generate Ground Truth (`generate_ground_truth.py`)

Script ini membaca label dari folder `labels/test/`, mengambil urutan karakter dari file `.txt` sesuai kelas dan posisi X, lalu menyusunnya menjadi **string plat nomor sebenarnya (ground truth)**. Hasil disimpan dalam file:

```
ground_truth.csv
```

Berisi dua kolom:
- `image` → nama file gambar
- `ground_truth` → plat nomor sebenarnya

---

### 2. OCR dengan LLaVA 

Script ini melakukan:

- Membaca `ground_truth.csv`
- Mengambil gambar dari dataset
- Mengirim permintaan OCR ke **model LLaVA** via **LM Studio API**
- Menerima hasil prediksi
- Menyimpan hasil prediksi dalam file:

```
ocr_result.csv
```

Berisi kolom:
- `image`
- `ground_truth`
- `prediction`
- `CER_score` → hasil evaluasi

> Model LLaVA yang digunakan: `llava-llama-3-8b-v1_1`

---

### 3. Evaluasi CER (`evaluate_cer.py`)

Script ini:

- Membaca file `ocr_result.csv`
- Menghitung **Character Error Rate (CER)** berdasarkan perbandingan `ground_truth` vs `prediction`

Formula CER:

```
CER = (S + D + I) / N
```

Keterangan:
- **S**: Substitusi karakter salah
- **D**: Deletion (karakter hilang)
- **I**: Insertion (karakter tambahan)
- **N**: Jumlah karakter ground truth

---

## Cara Menjalankan

1. **Jalankan LM Studio**
   - Pastikan model `llava-llama-3-8b-v1_1` sudah dimuat
   - Aktifkan server dengan perintah:
     ```
     lms server start
     ```

2. **Generate Ground Truth**
   ```bash
   py generate_ground_truth.py
   ```

3. **Lakukan OCR**
   ```bash
   py predict_ocr.py
   ```

4. **Evaluasi CER**
   ```bash
   py evaluate_cer.py
   ```

---

## Hasil

Hasil akhir akan disimpan dalam file:
- `ocr_result.csv` → lengkap dengan prediksi dan skor CER untuk setiap gambar.

---

## Lisensi

Proyek ini menggunakan dataset publik dari Kaggle dan bebas digunakan untuk keperluan riset dan pembelajaran.
