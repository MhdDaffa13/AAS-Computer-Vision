import os
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "Indonesian License Plate Recognition Dataset/images/test")
LABEL_DIR = os.path.join(BASE_DIR, "Indonesian License Plate Recognition Dataset/labels/test")
CLASS_FILE = os.path.join(BASE_DIR, "Indonesian License Plate Recognition Dataset/classes.names")
OUTPUT_CSV = "ground_truth.csv"

with open(CLASS_FILE, "r", encoding="utf-8")as f:
    classes = [line.strip() for line in f.readlines()]

image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image", "ground_truth"])  # header

    for image_file in image_files:
        name_wo_ext = os.path.splitext(image_file)[0]
        label_path = os.path.join(LABEL_DIR, name_wo_ext + ".txt")

        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as lf:
                label_lines = lf.readlines()

            # Ambil index kelas (digit pertama di setiap baris), urutkan berdasarkan posisi X jika tersedia
            labels = []
            for line in label_lines:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_idx = int(parts[0])
                    labels.append((float(parts[1]) if len(parts) > 1 else 0, classes[class_idx]))

            # Urutkan label berdasarkan posisi X (jika tersedia), agar karakter tersusun rapi
            labels.sort(key=lambda x: x[0])
            plate = "".join([char for _, char in labels])
        else:
            plate = ""

        writer.writerow([image_file, plate])
        
print(f"✅ ground_truth.csv berhasil dibuat di: {OUTPUT_CSV}")