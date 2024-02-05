import cv2
import YOLOdetector as yl
import Kalman as kalm
import numpy as np

detector = yl.object_detector()
cap = cv2.VideoCapture(0)

KF = kalm.KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)
x, y, x1, y1 = 0, 0, 0, 0

while True:
    _, img = cap.read()

    center = detector.detect(img)

    if (len(center)>0):

        cv2.circle(img, (int(center[0][0]), int(center[0][1])), radius=20, 
                   color=(0, 0, 255), thickness=-1)

        x, y = KF.predict()

        (x1, y1) = KF.filter(center[0])

    x, y = KF.predict()
    cv2.circle(img, (int(x), int(y)), 20, (255, 0, 0), -1)
    cv2.circle(img, (int(x1), int(y1)), 20, (0, 0, 0), -1)
    cv2.imshow("Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
