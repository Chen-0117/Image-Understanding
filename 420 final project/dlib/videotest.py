# get idea from https://github.com/davisking/dlib-models
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import cv2
import shot_detect
# construct the argument parser and parse the arguments
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

def has_face(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(img, width=750)
    boxes = face_recognition.face_locations(rgb)
    if boxes:
        return True
    return False

def inShot(index, l):
    for item in l:
        if item[0] <= index <= item[1] and (has_face(video_frames[item[0]]) or has_face(video_frames[item[1]])):
            return True
    return False

shotBoundaryIndex = shot_detect.shot_detector(args["video"])

i = 0
# loop over frames from the video file stream
while True:
  # grab the frame from the threaded video stream
  ret, frame = vs.read()
  if not ret:
    break
  # convert the input frame from BGR to RGB then resize it to have
  # a width of 750px (to speedup processing)
  if inShot(i, shotBoundaryIndex):
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
      cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
  i += 1
  cv2.imshow('img',frame)
  key = cv2.waitKey(5)
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
