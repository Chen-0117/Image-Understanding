#get idea from https://sabbir.dev/article/face-detection-with-opencv-haar-cascade-vs-dlib-hog/
import cv2
import numpy as np
import shot_detect

videoName = 'video/Movie_1.mp4'
lbph_recognizer = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8, 123)
eigen_recognizer = cv2.face.EigenFaceRecognizer_create(10)
fisher_recognizer = cv2.face.FisherFaceRecognizer_create(10)
lbph_recognizer.read('haarcascades/trainner/lbph_trainner.yml')
eigen_recognizer.read('haarcascades/trainner/eigen_trainner.yml')
fisher_recognizer.read('haarcascades/trainner/fisher_trainner.yml')


cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
shotBoundaryIndex = shot_detect.shot_detector(videoName)
# font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1) # in OpenCV 2
font = cv2.FONT_HERSHEY_SIMPLEX
video = cv2.VideoCapture(videoName)

# index_list = []
# for item in shotBoundaryIndex:
#     index_list.append(item[0])
#     index_list.append(item[1])
# print(shotBoundaryIndex)
video_frames = []
while True:
    # Read the next frame
    ret, frame = video.read()
    if frame is not None:
      video_frames.append(frame)
    # If there are no more frames, break out of the loop
    if not ret:
        break
video.release()
cv2.destroyAllWindows()

def has_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(15,15))
    if type(faces) == np.ndarray:
        return True
    return False

def inShot(index, l):
    for item in l:
        if item[0] <= index <= item[1] and (has_face(video_frames[item[0]]) or has_face(video_frames[item[1]])):
            return True
    return False

video1 = cv2.VideoCapture(videoName)
i = 0
while True:
    ret, img = video1.read()
    if not ret:
        break
    if inShot(i, shotBoundaryIndex):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(15,15))
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            # k1 = np.array([[y, x],[y+h,x],[y,x+w],[y+h,x+w]])
            # k2 = np.array([[0,0],[0,200],[200,0],[200,200]])
            # h,mark = cv2.findHomography(k1,k2,cv2.RANSAC)
            # face_test = cv2.warpPerspective(face, h, (200,200))
            face_test = cv2.resize(face, (200, 200))
            cv2.rectangle(img, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
            # img_id, conf = fisher_recognizer.predict(face_test)
            # if conf > 80:
            #     if img_id == 1:
            #         img_id = 'Thomas Kretschman'
            #     elif img_id == 2:
            #         img_id = 'Harrison Ford'
            #     elif img_id == 3:
            #         img_id = 'Phoebe Waller Bridge'
            #     # elif img_id == 4:
            #     #     img_id = 'Shaunette Renée Wilson'
            # else:
            #     img_id = "Unknown"
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
            cv2.putText(img, str(name), (x, y + h), font, 0.55, (0, 255, 0), 1)
    i += 1
    cv2.imshow('img',img)
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

    # if cv2.waitKey(100) & 0xFF == ord('q'):
    #     break
video1.release()
cv2.destroyAllWindows()