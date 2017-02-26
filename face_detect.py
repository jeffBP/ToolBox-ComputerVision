""" Experiment with face detection and image filtering using OpenCV """

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
kernel = np.ones((30, 30), 'uint8')
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20, 20))

    for (x, y, w, h) in faces:
        frame[y:y+h, x:x+w, :] = cv2.dilate(frame[y:y+h, x:x+w, :], kernel)
        cv2.circle(frame, center=(x + w/3, y + h/3), radius=20, color=(255, 255, 255), thickness=20)
        cv2.circle(frame, center=(x + w/3+5, y + h/3+5), radius=5, color=(0, 0, 0), thickness=20)
        cv2.circle(frame, center=(x + w - w/3, y + h/3), radius=20, color=(255, 255, 255), thickness=20)
        cv2.circle(frame, center=(x + w - w/3-5, y + h/3-5), radius=5, color=(0, 0, 0), thickness=20)
        #cv2.ellipse(frame, cv2.ellipse2Poly((x + w//2, y + 2*h//3), (w//4, h//8), 0, 180, 255, 1), color=(255, 255, 255), thickness=5)
    # Display the resulting frame-by-frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the Capture
cap.release()
cv2.destroyAllWindows()
