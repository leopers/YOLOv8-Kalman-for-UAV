import YOLOdetector as yl
import cv2

detector = yl.object_detector()
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    detector.detect(img)

    center_x, center_y = detector.center

    img = cv2.circle(img, (center_x, center_y), radius=20, color=(0, 0, 255), thickness=-1)
    
    cv2.imshow("Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
