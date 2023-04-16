import numpy as np
import cv2
# detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
sampleNum = 1
Id = 0
pic = 1

while True:
    if sampleNum > 45:
        sampleNum -= 45
    if pic <= 45:
        Id = 1
    elif 45 < pic <= 90:
        Id = 2
    elif 90 < pic <= 135:
        Id = 3
    # elif 60 < pic <= 80:
    #     Id = 4
    if Id == 1:
        filename = './actor/thomas kretschmann.' + str(Id) + '.' + str(sampleNum) + '.jpeg'
    elif Id == 2:
        filename = './actor/harrison ford.' + str(Id) + '.' + str(sampleNum) + '.jpeg'
    elif Id == 3:
        filename = './actor/phoebe waller bridge.' + str(Id) + '.' + str(sampleNum) + '.jpeg'
    # elif Id == 4:
    #     filename = './actor/Shaunette Renée Wilson.' + str(Id) + '.' + str(sampleNum) + '.jpeg'
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(50,50))
    if type(faces) == np.ndarray:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            f = cv2.resize(gray[y:y+h, x:x+w], (200,200))
            # saving the captured face in the dataset folder
            cv2.imwrite("dataSet/Actor." + str(Id) + '.' + str(sampleNum) + ".jpg", f) 

        cv2.imshow('frame', img)
        # wait for 100 miliseconds
    sampleNum += 1
    pic += 1
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    if pic > 135:
        break

cv2.destroyAllWindows()