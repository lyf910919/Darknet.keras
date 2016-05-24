# Darknet.keras
This is a modified version of Darknet.keras, with some bugs fixed and YOLO Model support.

###Contributions
1. Modified DarkNet.py to load YOLO Model(rather than YOLO Tiny Model) weights files.
2. Modified RunTinyYOLO.py, which had a few bugs, to reproduce YOLO detection results in darknet.

###TODO
1. Building network structure by hand is tedious, should write code to do so automatically from cfg files
2. Implement training code

###The original README
This is a transfer from weights trained by [Darknet](http://pjreddie.com/darknet/) to [keras](http://keras.io/) with Theano backend.

Currently I only finish test process for [YOLO Tiny Model](https://github.com/pjreddie/darknet/blob/master/cfg/yolo-tiny.cfg).

Due to image preprocess difference bewteen my reimplementaion and Darknet's, the result has slightly difference.

To run the code:

1. Create three folders:  `weights, results, images`.
2. Put the images you want to detect in the folder `images`.
2. Download yolo-tiny.weights from [Darknet website](http://pjreddie.com/darknet/yolo/) and put it in the folder `weights`.
3. Run:
   ```
   python RunTinyYolo.py
   ```
The detection result will be saved in the results folder

Notes:
Make sure your have Theano and Keras installed
