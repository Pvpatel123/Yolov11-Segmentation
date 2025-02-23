from ultralytics import YOLO

class YOLOv11Segmentation:
    def __init__(self, model_path="yolo11n-seg.pt"):
        """Initialize the YOLO segmentation model."""
        self.model = YOLO(model_path)

    def segment(self, image):
        """Perform segmentation on an image using YOLO."""
        results = self.model(image)
        return results
