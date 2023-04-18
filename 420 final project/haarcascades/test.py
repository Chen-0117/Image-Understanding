import cv2
import numpy as np

lbph_recognizer = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8, 123)
eigen_recognizer = cv2.face.EigenFaceRecognizer_create(10)
fisher_recognizer = cv2.face.FisherFaceRecognizer_create(10)
lbph_recognizer.read('trainner/lbph_trainner.yml')
eigen_recognizer.read('trainner/eigen_trainner.yml')
fisher_recognizer.read('trainner/fisher_trainner.yml')

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
        face = gray[y:y+h, x:x+w]
        face_test = cv2.resize(face, (200, 200))
        cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
        img_id, conf = lbph_recognizer.predict(gray[y:y + h, x:x + w])
        predictPCA = 0
        if not face_test is None:
            predictPCA, conf = eigen_recognizer.predict(face_test)
        if conf > 90:
                if predictPCA == 1:
                    name = "Thomas Kretschman"
                elif predictPCA == 2:
                    name = "Harrison Ford"
                elif predictPCA == 3:
                    name = "Phoebe Waller Bridge"
                elif predictPCA == 4:
                    name = "Michelle Yeoh"
                elif predictPCA == 5:
                    name = "Henry Golding"
                elif predictPCA == 6:
                    name = "Constance Wu"
                elif predictPCA == 7:
                    name = "Jonathan Pryce"
                elif predictPCA == 8:
                    name = "Max Irons"
                elif predictPCA == 9:
                    name = "Gleen Close"
        else:
                name = 'unknown'
        # cv2.cv.PutText(cv2.cv.fromarray(im), str(Id), (x, y + h), font, 255)
        cv2.putText(im, name, (x, y + h), font, 0.55, (0, 255, 0), 1)
    cv2.imshow('im', im)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()