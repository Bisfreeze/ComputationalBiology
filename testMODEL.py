from ultralytics import YOLO
import os

model = YOLO("D:\\BINUS File\\ComBio_AI_V1.1\\runs\\classify\\train13\\weights\\best.pt")  # masukin current train best.pt modelnya

# Predict with the model
results = model("melamela.jpg") 

print("\n" + "="*50)
print("SKIN CONDITION ANALYSIS RESULTS")
print("="*50)
print(f"Image: {os.path.basename(results[0].path)} ({results[0].orig_shape[0]}x{results[0].orig_shape[1]})")
print("-"*50)
print("PREDICTIONS (Ranked by Confidence):")
print("-"*50)

# Extract and sort predictions by confidence
predictions = []
for i, (cls, conf) in enumerate(zip(results[0].names.values(), results[0].probs.data.tolist())):
    if conf > 0.005:  # Only show predictions with confidence > 0.5%
        predictions.append((cls, conf*100))

# Sort by confidence (highest first)
predictions.sort(key=lambda x: x[1], reverse=True)

# Print predictions in a readable format
for i, (cls, conf) in enumerate(predictions):
    print(f"{i+1}. {cls}: {conf:.2f}%")

print("-"*50)
print(f"Processing Time: {results[0].speed['preprocess']:.1f}ms preprocess, "
      f"{results[0].speed['inference']:.1f}ms inference, "
      f"{results[0].speed['postprocess']:.1f}ms postprocess")
print("="*50 + "\n")