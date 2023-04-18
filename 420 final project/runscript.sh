#!/bin/bash
pip install -r requirements.txt
cd dlib
cd “dlib-19.24”
python setup.py install
pip install face_recognition
# python videotest.py --encodings encodings.pickle --video ../video/Movie_3.mp4