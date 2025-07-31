import os
import csv
import json
import base64
import difflib
import requests
from tqdm import tqdm

# === Konfigurasi ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Indonesian License Plate Recognition Dataset", "images", "test")
GROUND_TRUTH_CSV = os.path.join(BASE_DIR, "ground_truth.csv")
OUTPUT_CSV = "ocr_result.csv"

SERVER_URL = "http://localhost:1234/v1/chat/completions"
VLM_MODEL_NAME = "llava-llama-3-8b-v1_1"  

# === Fungsi Hitung CER ===
def calculate_cer(ground_truth, prediction):
    matcher = difflib.SequenceMatcher(None, ground_truth, prediction)
    S = D = I = 0
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == 'replace':
            S += max(i2 - i1, j2 - j1)
        elif opcode == 'delete':
            D += i2 - i1
        elif opcode == 'insert':
            I += j2 - j1
    N = len(ground_truth)
    CER = round((S + D + I) / N, 4)
    return f"{CER * 100:.2f}%"

# === Load Ground Truth ===
def load_ground_truth(csv_path):
    gt_dict = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_dict[row["image"]] = row["ground_truth"]
    return gt_dict

# === Encode Gambar ke base64 ===
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# === Query ke LLaVA ===
def ocr_image(image_path):
    try:
        image_base64 = encode_image_to_base64(image_path)
        image_url = f"data:image/jpeg;base64,{image_base64}"

        payload = {
            "model": VLM_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        },
                        {
                            "type": "text",
                            "text": "What is the license plate number shown in this image? Respond only with the plate number."
                        }
                    ]
                }
            ],
            "temperature": 0.2,
            "stream": False
        }

        response = requests.post(SERVER_URL, json=payload)
        response.raise_for_status()

        result = response.json()["choices"][0]["message"]["content"]
        return result.strip().replace(" ", "").upper()
    
    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {str(e)}")
        return "ERROR"

# === Main ===
def main():
    ground_truths = load_ground_truth(GROUND_TRUTH_CSV)
    if not ground_truths:
        print("No ground truth data loaded. Exiting.")
        return

    image_files = list(ground_truths.keys())

    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "ground_truth", "prediction", "CER_score"])

        for image_name in tqdm(image_files, desc="Processing Plates"):
            image_path = os.path.join(DATASET_DIR, image_name)
            gt_text = ground_truths[image_name]
            pred_text = ocr_image(image_path)
            cer = calculate_cer(gt_text, pred_text)
            writer.writerow([image_name, gt_text, pred_text, cer])
            print(f"\n{image_name} => GT: {gt_text} | pred: {pred_text} | CER: {cer}\n")

    print(f"\n OCR selesai. Hasil disimpan di {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
