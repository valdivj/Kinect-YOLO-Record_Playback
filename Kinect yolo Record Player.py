import cv2
from darkflow.net.build import TFNet
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np
import time
import tensorflow as tf
from eip import PLC
import pickle


#Uncoment next 2 lines for PLC support
# I am pushing data to a PLC running CLX 5000 software
#test = PLC()
#test.IPAddress = "172.16.2.161"


config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    options = {
            'model': 'cfg/yolov2-tiny-voc.cfg',
            'load': 'bin/yolov2-tiny-voc.weights',
            'threshold': 0.2,
            'gpu': 1.0
                    }
    tfnet = TFNet(options)

colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
KinectC = cv2.VideoCapture('C:/Users/shirley/Desktop/Kinect Yolo Recordplayback/Kinect_Color.mp4')
filename = 'Kinect_Depth'
infile = open(filename,'rb')
frame = pickle.load(infile)
frame_idx = 0

if KinectC.isOpened() == False:
    print(
        "Error opening the video file. Please double check your file path for typos. Or move the movie file to the same location as this script/notebook")

# While the video is opened
while KinectC.isOpened():

    # Read the video file.
    ret, frameC = KinectC.read()


    stime = time.time()
    if ret == True:
        frameC = cv2.resize(frameC, (0, 0), fx=0.5, fy=0.5)
        frameD = frame[frame_idx]
        depthxy = frame[frame_idx]
        frame_idx += 1
        if frame_idx == len(frame):
            frame_idx = 0
        depthxy = np.reshape(depthxy, (424, 512))
        frameD = frameD.astype(np.uint8)
        frameD = np.reshape(frameD, (424, 512))
        frameD = cv2.cvtColor(frameD, cv2.COLOR_GRAY2BGR)


        def click_eventD(event, x, y, flags, param):
           if event == cv2.EVENT_LBUTTONDOWN:
               print(x, y)
           if event == cv2.EVENT_RBUTTONDOWN:
               Pixel = depthxy[y]
               Pixel_Depth = Pixel[x]
               print(Pixel_Depth)

        def click_eventC(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(x, y)
            if event == cv2.EVENT_RBUTTONDOWN:
                red = frameC[y, x, 2]
                blue = frameC[y, x, 0]
                green = frameC[y, x, 1]
                print(red, green, blue)

        results = tfnet.return_predict(frameC)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            x_Center = int((((result['topleft']['x']) + (result['bottomright']['x'])) / 2))
            y_Center = int((((result['topleft']['y']) + (result['bottomright']['y'])) / 2))
            Center = (int(x_Center / 2), int(y_Center * .8))
            Pixel = depthxy[int(y_Center * .8)]
            Pixel_Depth = Pixel[int(x_Center / 2)]
            label = result['label']
            confidence = result['confidence']
            text = '{}:{:.0f}%'.format(label, confidence * 100)
            textD = 'Depth{}mm'.format(Pixel_Depth)
            frameC = cv2.rectangle(frameC, tl, br, color, 5)
            frameD = cv2.circle(frameD, Center, 10, color, -1)
            frameC = cv2.putText(frameC, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            frameC = cv2.putText(frameC, textD, br, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('frameC', frameC)
        cv2.imshow('frameD', frameD)
        cv2.setMouseCallback('frameC', click_eventC)
        cv2.setMouseCallback('frameD', click_eventD)
        # uncomment next 2 lines for PLC support
        # Make a String tag(YOLO_Sting) and a INT tag( YOLO_INT) in your CLX 5000 processor
       # ex: test.Write("YOLO_String", label)
       # ex: test.Write("YOLO_INT", Pixel_Depth)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
       #Un coment line below for PLC support
       #test.Close()q
    else:
        break
cv2.destroyAllWindows()
