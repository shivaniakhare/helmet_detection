
import cv2
from ultralytics import YOLO


model = YOLO("helmet.pt")


img = cv2.imread("helmet.jpg")
img = cv2.resize(img, (600, 400))


results = model(img, verbose=False)
class_names = model.names
count = 0


for result in results:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        continue

    for box in boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        label = class_names[cls_id]

        if conf > 0.5:
            count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{label} #{count}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


cv2.putText(img, f"Total Helmets: {count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


cv2.imshow("Detected Image", img)
cv2.imwrite("output_detected.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
