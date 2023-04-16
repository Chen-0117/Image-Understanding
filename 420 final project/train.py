import cv2
import os
import numpy as np
from PIL import Image
# detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
lbph_recognizer = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8, 123)
eigen_recognizer = cv2.face.EigenFaceRecognizer_create(10)
fisher_recognizer = cv2.face.FisherFaceRecognizer_create(10)

# recognizer = cv2.createLBPHFaceRecognizer()


def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        if image_path == 'dataSet/' + '.DS_Store':
            continue
        image = Image.open(image_path).convert('L')
        image_np = np.array(image, 'uint8')
        if os.path.split(image_path)[-1].split(".")[-1] != 'jpg':
            continue
        image_id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = detector.detectMultiScale(image_np, 1.1, 5, minSize=(50,50))
        for (x, y, w, h) in faces:
            face_samples.append(cv2.resize(image_np[y:y + h, x:x + w],(200,200)))
            ids.append(image_id)

    return face_samples, ids


faces, Ids = get_images_and_labels('dataSet')
lbph_recognizer.train(faces, np.array(Ids))
eigen_recognizer.train(faces, np.array(Ids))
fisher_recognizer.train(faces, np.array(Ids))
lbph_recognizer.save('trainner/lbph_trainner.yml')
eigen_recognizer.save('trainner/eigen_trainner.yml')
fisher_recognizer.save('trainner/fisher_trainner.yml')