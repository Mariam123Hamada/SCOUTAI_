from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict, Tuple

class ConeDetection:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self):
        return YOLO(self.model_path)

    def get_detections(self, frame, imgsz=320, conf=0.10):
        """Runs inference on a frame and returns detection results."""
        results = self.model(frame, imgsz=imgsz, conf=conf)
        return results

    def get_cone_centers(self, frame) -> List[Tuple[int,int]]:
        """Returns only cone center coordinates [(cx, cy), ...]."""
        results = self.get_detections(frame)[0]
        centers = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # skip football
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            centers.append((cx, cy))
        return centers
    def get_fotball_centers(self, frame) -> List[Tuple[int,int]]:
        """Returns only Football center coordinates [(cx, cy), ...]."""
        results = self.get_detections(frame)[0]
        centers = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 1:  # skip Cone
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            centers.append((cx, cy))
        return centers
    def get_detections_with_colors(self, frame) -> Dict[Tuple[int,int], str]:
        """Returns all detected objects with label and center."""
        results = self.get_detections(frame)[0]
        detections_dict = {}
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            label = "Football" if cls_id == 0 else "Cone"
            detections_dict[(cx, cy)] = label
        return detections_dict
