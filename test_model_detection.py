from ultralytics import YOLO
import os

# Load model
model = YOLO("runs/detect/train7/weights/best.pt")  # Ganti path sesuai modelmu

# Load image
results = model("images.jpg")  # Ganti dengan path gambar input kamu

print("\n" + "="*50)
print("SKIN CONDITION ANALYSIS RESULTS")
print("="*50)
print(f"Image: {os.path.basename('images.jpg')} ({results[0].orig_shape[0]}x{results[0].orig_shape[1]})")
print("\nPREDICTIONS (Ranked by Confidence):")

predictions = []

# Cek apakah ada box hasil deteksi
if results[0].boxes is not None:
    boxes = results[0].boxes
    for i in range(len(boxes.cls)):
        cls_id = int(boxes.cls[i].item())
        conf = boxes.conf[i].item()
        if conf >= 0.005:  # threshold bisa kamu sesuaikan
            predictions.append((cls_id, conf * 100))  # confidence jadi persen

    # Urutkan berdasarkan confidence tertinggi
    predictions.sort(key=lambda x: x[1], reverse=True)

    for i, (cls_id, conf) in enumerate(predictions):
        class_name = results[0].names[cls_id]
        print(f"{i+1}. {class_name}: {conf:.2f}%")
else:
    print("❌ No detections found.")

# Print inference time info
print("\n" + "="*50)
print(f"Processing Time: {results[0].speed['preprocess']:.1f}ms preprocess, "
      f"{results[0].speed['inference']:.1f}ms inference, "
      f"{results[0].speed['postprocess']:.1f}ms postprocess")
print("="*50 + "\n")
