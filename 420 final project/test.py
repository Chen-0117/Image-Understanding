import cv2
import numpy as np

lbph_recognizer = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8, 123)
lbph_recognizer.read('trainner/lbph_trainner.yml')

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
cam = cv2.VideoCapture('trailer3.mp4')
# font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1) # in OpenCV 2
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, im = cam.read()
    if not ret:
        break
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(15,15))
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
        img_id, conf = lbph_recognizer.predict(gray[y:y + h, x:x + w])
        if conf > 60:
            if img_id == 1:
                img_id = 'Thomas Kretschman'
            elif img_id == 2:
                img_id = 'Harrison Ford'
            elif img_id == 3:
                img_id = 'Phoebe Waller Bridge'
            # elif img_id == 4:
            #     img_id = 'Shaunette Renée Wilson'
        else:
            img_id = "Unknown"
        # cv2.cv.PutText(cv2.cv.fromarray(im), str(Id), (x, y + h), font, 255)
        cv2.putText(im, str(img_id), (x, y + h), font, 0.55, (0, 255, 0), 1)
    cv2.imshow('im', im)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()