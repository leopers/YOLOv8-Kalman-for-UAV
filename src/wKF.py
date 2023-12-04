import cv2
import YOLOdetector as yl
import Kalman as kalm

detector = yl.object_detector()
cap = cv2.VideoCapture(0)

KF = kalm.KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)
x ,y, x1, y1 = 0, 0, 0, 0

while True:
    _, img = cap.read()

    center = detector.detect(img)
    
    if center!=None:
        cv2.circle(img, center, radius=20, color=(0, 0, 255), thickness=-1)

        predicted_center = KF.predict()
        x, y = predicted_center[0][0].astype(int), predicted_center[1][0].astype(int)
        cv2.circle(img, (int(x), int(y)), radius=20, color=(255, 0, 0), thickness=-1)

        filtered_center = KF.filter(center)
        x1, y1 = filtered_center[0][0].astype(int), filtered_center[1][0].astype(int)
        cv2.circle(img, (int(x1), int(y1)), radius=20, color=(0, 0, 0), thickness=-1)

    cv2.circle(img, (int(x), int(y)), radius=20, color=(255, 0, 0), thickness=-1)
    cv2.circle(img, (int(x1), int(y1)), radius=20, color=(0, 0, 0), thickness=-1)

    
    cv2.imshow("Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

