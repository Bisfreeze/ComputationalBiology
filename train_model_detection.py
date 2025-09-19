from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/train7/weights/last.pt")
    model.train(
        data="D:/BINUS File/2025 file/CompBio_AI_V.2.1 detection/skin_disease_dataset/skin_disease.yaml",
        epochs=100,
        imgsz=320,
        batch=16,
        device=0,
        hsv_h=0.03,
        hsv_s=0.6,
        hsv_v=0.5,
        mosaic=1.0,
        mixup=0.1,
        scale=0.5,
        translate=0.2,
        fliplr=0.5,
        freeze=19,
        workers=8,
    )