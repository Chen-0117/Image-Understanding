import cv2
import matplotlib.pyplot as plt
import numpy as np


def shot_detector(video_name):
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
