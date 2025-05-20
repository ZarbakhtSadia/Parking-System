import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

def initialize_detection():
    try:
        model = YOLO('yolov8s.pt')
        print("✅ YOLO model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        return None

    # Load class names
    try:
        with open("coco.txt", "r") as my_file:
            class_list = my_file.read().split("\n")
    except Exception as e:
        print(f"❌ Error loading class names: {e}")
        return None

    # Define parking areas (same as original code)
    areas = [
        [(52,364),(30,417),(73,412),(88,369)],
        [(105,353),(86,428),(137,427),(146,358)],
        [(159,354),(150,427),(204,425),(203,353)],
        [(217,352),(219,422),(273,418),(261,347)],
        [(274,345),(286,417),(338,415),(321,345)],
        [(336,343),(357,410),(409,408),(382,340)],
        [(396,338),(426,404),(479,399),(439,334)],
        [(458,333),(494,397),(543,390),(495,330)],
        [(511,327),(557,388),(603,383),(549,324)],
        [(564,323),(615,381),(654,372),(596,315)],
        [(616,316),(666,369),(703,363),(642,312)],
        [(674,311),(730,360),(764,355),(707,308)]
    ]
    
    area_polygons = [np.array(area, np.int32) for area in areas]
    return model, class_list, area_polygons

def process_frame(frame, model, class_list, area_polygons):
    total_slots = len(area_polygons)
    frame_count = 0
    occupancy = [0] * total_slots

    # YOLO Prediction
    results = model.predict(frame, verbose=False)
    
    try:
        detections = results[0].boxes.data
        px = pd.DataFrame(detections).astype("float")

        for index, row in px.iterrows():
            x1, y1, x2, y2, _, cls_id = map(int, row)
            c = class_list[cls_id]

            if 'car' in c:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                for i in range(total_slots):
                    if cv2.pointPolygonTest(area_polygons[i], (cx, cy), False) >= 0:
                        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                        occupancy[i] = 1
                        break
        return frame, occupancy
    except Exception as e:
        print(f"❌ Processing error: {e}")
        return frame, occupancy