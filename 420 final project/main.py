import cv2
import matplotlib.pyplot as plt
import numpy as np


class VideoRecognize:
    def __init__(self, videoName):
        self.video = cv2.VideoCapture(videoName)
    
