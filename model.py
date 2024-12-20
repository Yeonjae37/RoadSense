import cv2
import torch
import numpy as np
import pyttsx3
import time
from fastai.vision.all import *
from datetime import datetime, timedelta
from pathlib import Path

path = Path('/Users/yeonjae-jeong/Documents/Weather/dataset')

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(224)
)

dls = dblock.dataloaders(path)

learn = vision_learner(dls, 'efficientnet_b3', metrics=[accuracy, error_rate], path='.').to_fp16()
learn.load("/Users/yeonjae-jeong/Documents/Weather/models/models")

engine = pyttsx3.init()

def speak_warning(message):
    engine.say(message)
    engine.runAndWait()

dangerous_weather_warnings = {
    #'fogsmog': ("짙은 안개로 가시거리가 짧습니다. 속도를 줄이고 주의 깊게 운전하세요.", 20),
    'glaze': ("도로가 빙판으로 변했습니다. 속도를 줄이고 급제동을 피하세요.", 10),
    'rain': ("폭우로 도로가 미끄럽습니다. 속도를 줄이고 안전 거리를 유지하세요.", 10),
    'hail': ("우박이 내리고 있습니다. 안전한 곳에 차량을 주차하세요.", 20),
    #'sandstorm': ("모래폭풍이 발생 중입니다. 속도를 줄이고 가시거리에 주의하세요.", 20),
    'snow': ("도로에 눈이 쌓였습니다. 속도를 줄이고 안전 거리를 확보하세요.", 10),
    'rime': ("도로에 서리가 끼었습니다. 속도를 줄이고 주의 깊게 운전하세요.", 10),
}

def predict_weather_from_frame(frame):
    img = PILImage.create(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize((224, 224))
    pred, _, probs = learn.predict(img)
    return pred, probs.max().item()

def is_dangerous_weather(pred):
    return pred in dangerous_weather_warnings

yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', 
                            path='/Users/yeonjae-jeong/Documents/PotholeDetector/yolov5/runs/train/pothole_yolov5s_results/weights/best.pt')
yolo_model.eval()

model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

today = datetime.today().strftime('%Y-%m-%d')
save_dir = Path(f'/Users/yeonjae-jeong/Documents/PotholeDetector/output{today}')
save_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()
    
last_warning_time = {}
last_saved_time = None
frame_count = 0

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        depth_prediction = midas(input_batch)
        depth_prediction = torch.nn.functional.interpolate(
            depth_prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    depth_map = depth_prediction.cpu().numpy()
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX)
    depth_map_uint8 = (depth_map_normalized * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_map_uint8, cv2.COLORMAP_MAGMA)

    results = yolo_model(frame)
    pothole_boxes = results.xyxy[0].cpu().numpy()

    for box in pothole_boxes:
        if len(box) >= 6:
            x1, y1, x2, y2, conf, cls = box[:6]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if cls == 0 and conf >= 0.20:
                current_time = datetime.now()
                if last_saved_time is None or current_time - last_saved_time > timedelta(seconds=20):
                    save_path = save_dir / f'pothole_{frame_count}.jpg'
                    cv2.imwrite(str(save_path), frame)
                    print(f"포트홀 이미지 저장: {save_path}, 정확도: {conf:.2f}")
                    last_saved_time = current_time

            pothole_depth_map = depth_map[y1:y2, x1:x2]
            median_depth = np.median(pothole_depth_map)
            pothole_width = x2 - x1
            pothole_height = y2 - y1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"Depth: {median_depth:.2f}, Width: {pothole_width}px"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    pred, prob = predict_weather_from_frame(frame)
    weather_label = f'Prediction: {pred}, Confidence: {prob:.2f}'
    cv2.putText(frame, weather_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if is_dangerous_weather(pred) and prob >= 0.95:
        message, interval = dangerous_weather_warnings[pred]

        current_time = time.time()
        if pred not in last_warning_time or current_time - last_warning_time[pred] >= interval:
            speak_warning(message)
            last_warning_time[pred] = current_time

    cv2.imshow('Pothole Detection with Depth and Weather', frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
