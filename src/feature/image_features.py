import pymupdf4llm
import pymupdf
import pandas as pd
import cv2
from doclayout_yolo import YOLOv10

model = YOLOv10("artifacts/doclayout_yolo_docstructbench_imgsz1024.pt")

class_styles = {
    0: {'color': (255, 0, 0), 'alpha': 0.3},      # Bright Red
    1: {'color': (0, 255, 0), 'alpha': 0.3},      # Bright Green
    2: {'color': (0, 0, 255), 'alpha': 0.3},      # Bright Blue
    3: {'color': (255, 255, 0), 'alpha': 0.3},    # Cyan (Yellow in BGR)
    4: {'color': (255, 0, 255), 'alpha': 0.3},    # Magenta
    5: {'color': (0, 255, 255), 'alpha': 0.3},    # Yellow (Cyan in BGR)
    6: {'color': (255, 128, 0), 'alpha': 0.3},    # Orange
    7: {'color': (128, 0, 128), 'alpha': 0.3},    # Purple
    8: {'color': (0, 128, 128), 'alpha': 0.3},    # Teal
    9: {'color': (255, 255, 255), 'alpha': 0.3},  # White
}

def get_layout(document, pagenr=0):
  image = document.pages[pagenr].image
  resized = image.resize((1024,1024))
  det_res = model.predict(
      resized,   # Image to predict
      imgsz=1024,        # Prediction image size
      conf=0.2,          # Confidence threshold
      device="mps",    # Device to use (e.g., 'cuda:0' or 'cpu')
  )
  annotated_frame = det_res[0].orig_img.copy()

  for box in det_res[0].boxes:
      x1, y1, x2, y2 = map(int, box.xyxy[0])
      cl = int(box.cls[0])
      color = class_styles.get(cl, {'color': (0, 0, 0), 'alpha': 0.3})
      cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color['color'], -1)  # -1 = filled

  # alpha = 0.3
  # cv2.addWeighted(det_res[0].orig_img.copy(), alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

  return annotated_frame