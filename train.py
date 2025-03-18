from ultralytics import YOLO

if __name__ == "__main__":  # Ensure multiprocessing works on Windows
    # Load a model
    model = YOLO("yolo11x-cls.pt")  # Load a pretrained model (recommended for training)
    
    # Train the model
    results = model.train(
        data="D:\\BINUS File\\ComBio_AI_V1.1\\dataset",
        epochs=100,
        imgsz=288,
        device=0,
        batch=32,
        hsv_h=0.05,  
        hsv_s=0.7,  
        hsv_v=0.4,  
        degrees=45,  
        fliplr=0.5,
        bgr=0.1,
        crop_fraction=0.8, 
    )