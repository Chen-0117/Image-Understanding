# Please read this introduction before testing my code.

Because there is a limit of 10 MB each file submitted on Markus, the code is untidy and missing which may lead to the code not running. I hope you can download the code from my github. 

Here is my github link: **https://github.com/Chen-0117/Image-Understanding**. Since the commit date will be recorded, there is no problem about late submission. Thanks in advance.

## There are three files I am unable to submit on Markus:

> * I cannot submit "actor" folder which is my training dataset. It won't make any difference to the final > result, but you can not run "haarcascades/train.py" and "dlib/dnncreate.py".
>
> * I cannot submit "dlib-19.24" which is helpful for you to install dlib. The normal way for dlib install is "pip install dlib". However, if it doesn't work, you can try “python setup.py install” if you cd into the "dlib-19.24" folder.
> * I cannot submit "trainner" folder which contains the trainning file for Haar Cascade Classifiler. I have submitted a zip file of it. Please unzip it before running the code.

## The elegant display of the file should be like below:

<img src="/Users/chen/Library/Application Support/typora-user-images/image-20230417221544565.png" alt="image-20230417221544565" style="zoom:50%;" />

> __pycache__
>
> > Environment file
>
> **actor**
>
> > A folder contains all the original training data
>
> **dataSet**
>
> > Training data set
>
> **dlib**
>
> > Dlib Classifier
> >
> > > **pycache**
> > >
> > > > Environment
> > >
> > > **dlib-19.24**
> > >
> > > > dlib installation folder
> > >
> > > **dnncreate.py**
> > >
> > > > A function to create training data set for dlib model
> > >
> > > **encodings.pickle**
> > >
> > > > Trained data
> > >
> > > **mmod_human_face_detector.dat**
> > >
> > > **phototest.py**
> > >
> > > > A function to test faces in photo
> > >
> > > **shot_detect.py**
> > >
> > > > A function to detect shot in videos
> > >
> > > **videotest.py**
> > >
> > > > A function to test faces from video shots and name them, write output to output video files
>
> **haarcascades**
>
> > Haar Cascasde Classifier
> >
> > >**haarcascade_frontalface_default.xml**
> > >
> > >> downloaded file for Haar Cascasde Classifier to detect faces
> > >
> > >**process.py**
> > >
> > >> A funciton to create trainning set
> > >
> > >**shot_detect.py**
> > >
> > >> A function to detect shot in videos
> > >
> > >**test.py**
> > >
> > >>A function to directly test faces and name them from video
> > >
> > >**testShot.py**
> > >
> > >> A funciton to test faces and name them from video shots, and add back to video
> > >
> > >**train.py**
> > >
> > >> A function to train LPH, Eigen, Fisher detector
> > >
> > >**trainner**
> > >
> > >> A folder contain all the trained ".xml" files
> > >
> > >**transform.py**
> > >
> > >> A funciton to transform the filename into tidy names
>
> **main.py**
>
> > An oop way to give a clean and tidy display of all the code.
>
> **markedVideo**
>
> > A folder contains video that are marked "shot #"
>
> **output_video**
>
> > A folder contains the result. (Video where inside all actor faces are detected and named)
>
> **readme.md**
>
> > A markdown file for instruction on the code
>
> **requirenment.txt**
>
> > Contains the environment you need
>
> **runscript.sh**
>
> > A bash script to run the requirement.txt
>
> **shot_accuracy.py**
>
> > A function to test the accuracy of shot_detect
>
> **shot_detect.py**
>
> > A function to detect different shot in the video
>
> **video**
>
> > A folder contains the three trailers we are going to test.
>
> **video_mark.py**
>
> > A function to mark the video(add "shot #" in the video)



## How to run the code

+ **./runscript.sh** (or **chmod +x ./runscript.sh** if that doesn't work)
+ Run dlib model

> cd into dlib file path

> python videotest.py --encodings encodings.pickle --video "videoname" --output "output video name"

> e.g. **python videotest.py --encodings encodings.pickle --video "../video/Movie_3.mp4" --output "../output_video/Movie3_output.avi"**

+ Run Haar Cascade Model

> cd into haarcascades file path
>
> python testShot.py

