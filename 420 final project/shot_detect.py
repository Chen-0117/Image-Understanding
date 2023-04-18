import cv2
import matplotlib.pyplot as plt
import numpy as np


def shot_detector(video_name):
    video = cv2.VideoCapture(video_name)

    # Set the threshold for shot detection
    threshold = 7
    # Initialize variables
    frame_count = 0
    prev_frame = None
    shots = []
    dif = []
    start = True
    skip = True

    while True:
        # Read the next frame
        ret, frame = video.read()

        # If there are no more frames, break out of the loop
        if not ret:
            if shots:
                shots[-1][-1] = frame_count
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If this is not the first frame, compare it to the previous frame
        if prev_frame is not None:

            diff = cv2.absdiff(gray_frame, prev_frame)
            # print(frame_count, np.average(diff))

            if np.average(diff) > 0.2 or not skip:
                if len(dif) > 0:
                    if np.average(diff) - dif[-1] > threshold:
                        shots[-1][-1] = frame_count - 1
                        shots.append([frame_count, frame_count])
                dif.append(np.average(diff))
                skip = True
            elif skip:
                shots[-1][-1] = frame_count
                skip = False

        else:
            shots.append([frame_count, frame_count])
            start = False

        # Update variables
        frame_count += 1
        prev_frame = gray_frame

    # Print the shot boundaries
    print(shots)
    return shots
