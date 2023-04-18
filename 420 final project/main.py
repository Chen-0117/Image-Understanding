import cv2
import numpy as np
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
from imutils import paths
import argparse
import os
from PIL import Image

class dlib:
    def __init__(self):
        self.video_frames = []

    def create(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--dataset", required=True,
        help="path to input directory of faces + images")
        ap.add_argument("-e", "--encodings", required=True,
        help="path to serialized db of facial encodings")
        ap.add_argument("-d", "--detection-method", type=str, default="cnn",
        help="face detection model to use: either `hog` or `cnn`")
        args = vars(ap.parse_args())

        # grab the paths to the input images in our dataset
        print("[INFO] quantifying faces...")
        imagePaths = list(paths.list_images(args["dataset"]))
        # initialize the list of known encodings and known names
        knownEncodings = []
        knownNames = []

        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(i + 1,
                len(imagePaths)))
            name = 'unknown'
            tag = int(imagePath.split(os.path.sep)[-1].split('.')[1])
            if tag == 1:
                name = "Thomas Kretschman"
            elif tag == 2:
                name = "Harrison Ford"
            elif tag == 3:
                name = "Phoebe Waller Bridge"
            elif tag == 4:
                name = "Michelle Yeoh"
            elif tag == 5:
                name = "Henry Golding"
            elif tag == 6:
                name = "Constance Wu"
            elif tag == 7:
                name = "Jonathan Pryce"
            elif tag == 8:
                name = "Max Irons"
            elif tag == 9:
                name = "Gleen Close"
            # name = imagePath.split(os.path.sep)[-2]
            # load the input image and convert it from BGR (OpenCV ordering)
            # to dlib ordering (RGB)
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input image
            boxes = face_recognition.face_locations(rgb,
                model=args["detection_method"])
            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)
            # loop over the encodings
            for encoding in encodings:
                # add each encoding + name to our set of known names and
                # encodings
                knownEncodings.append(encoding)
                knownNames.append(name)

        # dump the facial encodings + names to disk
        print("[INFO] serializing encodings...")
        data = {"encodings": knownEncodings, "names": knownNames}
        f = open(args["encodings"], "wb")
        f.write(pickle.dumps(data))
        f.close()

    def phototest(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-e", "--encodings", required=True,
        help="path to serialized db of facial encodings")
        ap.add_argument("-i", "--image", required=True,
        help="path to input image")
        ap.add_argument("-d", "--detection-method", type=str, default="cnn",
        help="face detection model to use: either `hog` or `cnn`")
        args = vars(ap.parse_args())
        # load the known faces and embeddings
        print("[INFO] loading encodings...")
        data = pickle.loads(open(args["encodings"], "rb").read())
        # load the input image and convert it from BGR to RGB
        image = cv2.imread(args["image"])
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # detect the (x, y)-coordinates of the bounding boxes corresponding
        # to each face in the input image, then compute the facial embeddings
        # for each face
        boxes = face_recognition.face_locations(rgb,
        model=args["detection_method"])
        encodings = face_recognition.face_encodings(rgb, boxes)
        # initialize the list of names for each face detected
        names = []
        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                encoding)
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                # determine the recognized face with the largest number of
                # votes (note: in the event of an unlikely tie Python will
                # select first entry in the dictionary)
                name = max(counts, key=counts.get)
            
            # update the list of names
            names.append(name)
        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)
        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)

    def shot_detector(self, video_name):
        video = cv2.VideoCapture(video_name)

        # Set the threshold for shot detection
        threshold = 6

        # Initialize variables
        frame_count = 0
        prev_frame = None
        shots = []
        dif = []
        start = True

        while True:
            # Read the next frame
            ret, frame = video.read()

            # If there are no more frames, break out of the loop
            if not ret:
                break

            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # If this is not the first frame, compare it to the previous frame
            # print(prev_frame)
            if prev_frame is not None:

                diff = cv2.absdiff(gray_frame, prev_frame)

                if len(dif) > 0:
                    # print(np.average(diff) - dif[-1] > threshold,start)
                    if dif[-1] - np.average(diff) > threshold and start:

                        shots.append([frame_count, frame_count])

                        # plt.imshow(prev_frame, cmap='gray')
                        # plt.axis("off")
                        # plt.title("start at " + str(frame_count))
                        # plt.show()
                        start = False
                    elif np.average(diff) - dif[-1] > threshold and not start:

                        # plt.imshow(prev_frame, cmap='gray')
                        # plt.axis("off")
                        # plt.title("end at " + str(frame_count))
                        # plt.show()


                        shots[-1][-1] = frame_count
                        start = True
                    dif.append(np.average(diff))

            else:
                shots.append([frame_count, frame_count])

                # plt.imshow(gray_frame, cmap='gray')
                # plt.axis("off")
                # plt.title("start at " + str(frame_count))
                # plt.show()

                start = False
                dif.append(0)

            # Update variables
            frame_count += 1
            prev_frame = gray_frame

        # Print the shot boundaries
        return shots

    def has_face(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(img, width=750)
        boxes = face_recognition.face_locations(rgb)
        if boxes:
            return True
        return False

    def inShot(self, index, l):
        for item in l:
            if item[0] <= index <= item[1] and (self.has_face(self.video_frames[item[0]]) or 
                                                self.has_face(self.video_frames[item[1]])):
                return True
        return False
    
    def videotest(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-e", "--encodings", required=True,
        help="path to serialized db of facial encodings")
        ap.add_argument("-v", "--video", required=True,
            help="path to input video file")
        ap.add_argument("-o", "--output", type=str,
        help="path to output video")
        # ap.add_argument("-y", "--display", type=int, default=1,
        #   help="whether or not to display output frame to screen")
        # ap.add_argument("-d", "--detection-method", type=str, default="cnn",
        #   help="face detection model to use: either `hog` or `cnn`")
        args = vars(ap.parse_args())
        # load the known faces and embeddings
        data = pickle.loads(open(args["encodings"], "rb").read())
        # initialize the video stream and pointer to output video file, then
        # allow the camera sensor to warm up
        vs = cv2.VideoCapture(args["video"])
        # vs = VideoStream(src=0).start()
        writer = None
        video = cv2.VideoCapture(args["video"])
        while True:
            # Read the next frame
            ret, frame = video.read()
            if frame is not None:
                self.video_frames.append(frame)
            # If there are no more frames, break out of the loop
            if not ret:
                break
        video.release()
        shotBoundaryIndex = self.shot_detector(args["video"])
        i = 0
        # loop over frames from the video file stream
        while True:
        # grab the frame from the threaded video stream
            ret, frame = vs.read()
            if not ret:
                break
        # convert the input frame from BGR to RGB then resize it to have
        # a width of 750px (to speedup processing)
            if self.inShot(i, shotBoundaryIndex):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = imutils.resize(frame, width=750)
                r = frame.shape[1] / float(rgb.shape[1])
                # detect the (x, y)-coordinates of the bounding boxes
                # corresponding to each face in the input frame, then compute
                # the facial embeddings for each face
                boxes = face_recognition.face_locations(rgb)
                encodings = face_recognition.face_encodings(rgb, boxes)
                names = []
            # loop over the facial embeddings
                for encoding in encodings:
                # attempt to match each face in the input image to our known
                # encodings
                    matches = face_recognition.compare_faces(data["encodings"],
                        encoding)
                    name = "Unknown"
                # check to see if we have found a match
                    if True in matches:
                        # find the indexes of all matched faces then initialize a
                        # dictionary to count the total number of times each face
                        # was matched
                        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                        counts = {}
                        # loop over the matched indexes and maintain a count for
                        # each recognized face face
                        for i in matchedIdxs:
                            name = data["names"][i]
                            counts[name] = counts.get(name, 0) + 1
                        # determine the recognized face with the largest number
                        # of votes (note: in the event of an unlikely tie Python
                        # will select first entry in the dictionary)
                        name = max(counts, key=counts.get)
                        if counts[name] <= 3 :
                            name = 'Unknown'
                
                    # update the list of names
                    names.append(name)
            # loop over the recognized faces
                for ((top, right, bottom, left), name) in zip(boxes, names):
                    # rescale the face coordinates
                    top = int(top * r)
                    right = int(right * r)
                    bottom = int(bottom * r)
                    left = int(left * r)
                    # draw the predicted face name on the image
                    cv2.rectangle(frame, (left, top), (right, bottom),
                        (0, 255, 0), 2)
                    y = top - 15 if top - 15 > 15 else top + 15
                    cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
            i += 1
            cv2.imshow('img',frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            # if the video writer is None *AND* we are supposed to write
            # the output video to disk initialize the writer
            if writer is None and args["output"] is not None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 20,
                (frame.shape[1], frame.shape[0]), True)
            # if the writer is not None, write the frame with recognized
            # faces to disk
            if writer is not None:
                writer.write(frame)
        # check to see if we are supposed to display the output frame to
        # the screen
        #   if args["display"] > 0:
        #     cv2.imshow("Frame", frame)
        #     key = cv2.waitKey(0) & 0xFF
        #     # if the `q` key was pressed, break from the loop
        #     if key == ord("q"):
        #       break
        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.release()
        # check to see if the video writer point needs to be released
        if writer is not None:
            writer.release()


class haarcascade:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        self.lbph_recognizer = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8, 123)
        self.eigen_recognizer = cv2.face.EigenFaceRecognizer_create(10)
        self.fisher_recognizer = cv2.face.FisherFaceRecognizer_create(10)
        self.videoName = 'video/Movie_2.mp4'
        self.cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        self.video_frames = []
    
    def procress(self):
        sampleNum = 1
        Id = 0
        pic = 1

        while True:
            if sampleNum > 14:
                sampleNum -= 14
            if pic <= 14:
                Id = 1
            elif 14 < pic <= 28:
                Id = 2
            elif 28 < pic <= 42:
                Id = 3
            elif 42 < pic <= 56:
                Id = 4
            elif 56 < pic <= 70:
                Id = 5
            elif 70 < pic <= 84:
                Id = 6
            elif 84 < pic <= 98:
                Id = 7
            elif 98 < pic <= 112:
                Id = 8
            elif 112 < pic <= 126:
                Id = 9
            if Id == 1:
                filename = './actor/thomas kretschmann.' + str(Id) + '.' + str(sampleNum) + '.jpeg'
            elif Id == 2:
                filename = './actor/harrison ford.' + str(Id) + '.' + str(sampleNum) + '.jpeg'
            elif Id == 3:
                filename = './actor/phoebe waller bridge.' + str(Id) + '.' + str(sampleNum) + '.jpeg'
            elif Id == 4:
                filename = './actor/Michelle Yeoh.' + str(Id) + '.' + str(sampleNum) + '.jpeg'
            elif Id == 5:
                filename = './actor/Henry Golding.' + str(Id) + '.' + str(sampleNum) + '.jpeg'
            elif Id == 6:
                filename = './actor/COnstance Wu.' + str(Id) + '.' + str(sampleNum) + '.jpeg'
            elif Id == 7:
                filename = './actor/Jonathan Pryce.' + str(Id) + '.' + str(sampleNum) + '.jpeg'
            elif Id == 8:
                filename = './actor/Max Irons.' + str(Id) + '.' + str(sampleNum) + '.jpeg'
            elif Id == 9:
                filename = './actor/Glenn Close.' + str(Id) + '.' + str(sampleNum) + '.jpeg'
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.3, 5, minSize=(50,50))
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

    def shot_detector(self, video_name):
        video = cv2.VideoCapture(video_name)

        # Set the threshold for shot detection
        threshold = 6

        # Initialize variables
        frame_count = 0
        prev_frame = None
        shots = []
        dif = []
        start = True

        while True:
            # Read the next frame
            ret, frame = video.read()

            # If there are no more frames, break out of the loop
            if not ret:
                break

            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # If this is not the first frame, compare it to the previous frame
            # print(prev_frame)
            if prev_frame is not None:

                diff = cv2.absdiff(gray_frame, prev_frame)

                if len(dif) > 0:
                    # print(np.average(diff) - dif[-1] > threshold,start)
                    if dif[-1] - np.average(diff) > threshold and start:

                        shots.append([frame_count, frame_count])

                        # plt.imshow(prev_frame, cmap='gray')
                        # plt.axis("off")
                        # plt.title("start at " + str(frame_count))
                        # plt.show()
                        start = False
                    elif np.average(diff) - dif[-1] > threshold and not start:

                        # plt.imshow(prev_frame, cmap='gray')
                        # plt.axis("off")
                        # plt.title("end at " + str(frame_count))
                        # plt.show()


                        shots[-1][-1] = frame_count
                        start = True
                    dif.append(np.average(diff))

            else:
                shots.append([frame_count, frame_count])

                # plt.imshow(gray_frame, cmap='gray')
                # plt.axis("off")
                # plt.title("start at " + str(frame_count))
                # plt.show()

                start = False
                dif.append(0)

            # Update variables
            frame_count += 1
            prev_frame = gray_frame

        # Print the shot boundaries
        return shots
    
    def train(self):
        faces, Ids = self.get_images_and_labels('dataSet')
        self.lbph_recognizer.train(faces, np.array(Ids))
        self.eigen_recognizer.train(faces, np.array(Ids))
        self.fisher_recognizer.train(faces, np.array(Ids))
        self.lbph_recognizer.save('haarcascades/trainner/lbph_trainner.yml')
        self.eigen_recognizer.save('haarcascades/trainner/eigen_trainner.yml')
        self.fisher_recognizer.save('haarcascades/trainner/fisher_trainner.yml')

    def get_images_and_labels(self,path):
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
            faces = self.detector.detectMultiScale(image_np, 1.1, 5, minSize=(50,50))
            for (x, y, w, h) in faces:
                face_samples.append(cv2.resize(image_np[y:y + h, x:x + w],(200,200)))
                ids.append(image_id)

        return face_samples, ids
    
    def createframe(self):
        video = cv2.VideoCapture(self.videoName)
        while True:
            # Read the next frame
            ret, frame = video.read()
            if frame is not None:
                self.video_frames.append(frame)
            # If there are no more frames, break out of the loop
            if not ret:
                break
        video.release()
    
    def has_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(15,15))
        if type(faces) == np.ndarray:
            return True
        return False

    def inShot(self, index, l):
        for item in l:
            if item[0] <= index <= item[1] and (self.has_face(self.video_frames[item[0]]) or
                                                 self.has_face(self.video_frames[item[1]])):
                return True
        return False

    def testshot(self):
        self.lbph_recognizer.read('haarcascades/trainner/lbph_trainner.yml')
        self.eigen_recognizer.read('haarcascades/trainner/eigen_trainner.yml')
        self.fisher_recognizer.read('haarcascades/trainner/fisher_trainner.yml')

        shotBoundaryIndex = self.shot_detector(self.videoName)
        # font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1) # in OpenCV 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        video = cv2.VideoCapture(self.videoName)
        v = cv2.VideoCapture(self.videoName)
        while True:
            # Read the next frame
            ret, frame = v.read()
            if frame is not None:
                self.video_frames.append(frame)
            # If there are no more frames, break out of the loop
            if not ret:
                break
        v.release()
        i = 0
        while True:
            ret, img = video.read()
            if not ret:
                break
            if self.inShot(i, shotBoundaryIndex):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(15,15))
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
                        predictPCA, conf = self.eigen_recognizer.predict(face_test)
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
        video.release()
        cv2.destroyAllWindows()

    

if __name__ == '__main__':
    h = haarcascade()
    h.testshot()
    # d = dlib()
    # d.videotest()