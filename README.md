# Kinect YOLO Record_Playback


 The next 3 lines are not my work,but you will need this info to setup YOLO on youre machine:

Real-time object detection and classification. Paper: [version 1](https://arxiv.org/pdf/1506.02640.pdf), [version 2](https://arxiv.org/pdf/1612.08242.pdf).

Read more about YOLO (in darknet) and download weight files [here](http://pjreddie.com/darknet/yolo/). In case the weight file cannot be found, I uploaded some of mine [here](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU), which include `yolo-full` and `yolo-tiny` of v1.0, `tiny-yolo-v1.1` of v1.1 and `yolo`, `tiny-yolo-voc` of v2.




## Dependencies

Python3, tensorflow 1.0, numpy, opencv 3. PYkinect2

 bin/tiny-yolo.weights --json
 
 These programs are based on the "Kinect2 YOLO Depth" repository. I needed a way to record the KINECT2 depth and video stream 
 So I could uploaded them to my papaerspace machine.
 
 1.kinect record.py : It records the video stream as a MP4 file and the depth stream as a pickle file.
 Make sure you set the path to the video and depth file locations.
 
 2.Kinect yolo Record Player.py : This plays back the video and depth stream files at the same time. The video stream is ran through the YOLO model and the objects are tagged with bounding boxes and labels. Then it uses the center of the bounding boxes to located the location of the center of the bounding box on the depth stream and marks and returns the depth measurement.
 
 You also can right and left click on the depth and video stream to return the X Y cordinates, depth measurment and RGB values of the clicked on pixel.
 
 
