from ultralytics import YOLO
import multiprocessing

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="data.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        pretrained=True,
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
