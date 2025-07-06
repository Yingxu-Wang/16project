# detect_inventory.py
import os
import json
import csv
import datetime
import time
import cv2
from ultralytics import YOLO

# —— 配置 ——
MODEL_PATH = "runs/detect/train12/weights/best.pt"  # 改成你的模型路径
CLASS_MAP = {0: "chips1", 1: "chips2", 2: "peanuts"}

BASE_DIR = os.path.dirname(__file__)
JSON_PATH = os.path.join(BASE_DIR, "inventory.json")
CSV_PATH  = os.path.join(BASE_DIR, "history.csv")

# 初始化 history.csv（如果不存在，写入表头）
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "chips1", "chips2", "peanuts"])

# 加载模型
model = YOLO(MODEL_PATH)

# 打开摄像头（或改成你 robomaster 的视频源）
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头，请检查设备")

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.1)
        continue

    # YOLO 推理
    results = model(frame, imgsz=640, conf=0.1, verbose=False)[0]

    # 统计各类别数量
    counts = {"chips1": 0, "chips2": 0, "peanuts": 0}
    for cls in results.boxes.cls:
        cls_id = int(cls.cpu().numpy())
        name = CLASS_MAP.get(cls_id)
        if name:
            counts[name] += 1

    # 写入 inventory.json
    try:
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(counts, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERR] 写入 inventory.json 失败：{e}")

    # 追加一行到 history.csv
    try:
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            ts = datetime.datetime.now().isoformat()
            writer.writerow([ts, counts["chips1"], counts["chips2"], counts["peanuts"]])
    except Exception as e:
        print(f"[ERR] 写入 history.csv 失败：{e}")

    # （可选）调试可视化
    # for box in results.boxes.xyxy:
    #     x1, y1, x2, y2 = map(int, box.cpu().numpy())
    #     cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    # cv2.imshow("Detect", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # 控制检查频率
    time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()
