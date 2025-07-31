# VideoTractor
This the example code for a simple video tractor. 

## Question

The goal is to build a video object tracker based on per-frame object category detections (i.e., 'tracking by detection'). 
The video of interest contains first-person-view footage from an RC car, with a 640$\times$360 pixels resolution. Starting from the coordinates $(x, y) = (566, 246)$, corresponding to the center point of one of the RC cars, the goal is to code a system that can follow it for as long as possible. 
The video can be downloaded at [HERE](www.robots.ox.ac.uk/~vgg/blog/rc-cars.zip).
The tracker is to be based on a pre-trained object detector. 
PyTorch and other popular frameworks already include a standard interface to download and run one with one or very few lines of code. 
No training with extra data is allowed – instead, you should code a simple strategy/heuristic to perform the tracking entirely online.



## Soulution
The goal is to track an RC car throughout a video based on frame-by-frame object detections. Since the object of interest is known at the start through a specific coordinate in the first frame, we employ a heuristic-based tracking-by-detection approach as follows:

- Use the given point ($x=566, y=246$) in the first frame to identify the corresponding object bounding box from the detected objects.
- For each subsequent frame, run a pre-trained object detector to identify all candidate objects. Associate the object with the tracked one from the previous frame using a heuristic strategy.
- Update the tracked bounding box and continue tracking frame by frame until the end of the video.
- Overlay the tracking results on the video for qualitative evaluation.

## Run the Code.

Firstly, install all necessary dependencies using pip:
```
pip install -r requirements.txt
```
or manually install:
```
pip install torch torchvision opencv-python numpy scipy filterpy
```

Secondly, set configure parameters  the parameters such as the input video frames path, detector model name, and initial tracking point. These can be set in the script:
```
FRAME_DIR        = './rc-cars/rc-cars'         # video frames directory
INIT_CENTER_PT   = (566, 246)                # starting coordinates
FRAME_SIZE       = (640, 360)                # pixels resolution
OBJECT_DETECTOR  = 'fasterrcnn_resnet50_fpn'
OUTPUT_VIDEO     = "tracked_rc_car.mp4"      # output video file
FPS              = 30                        # frame rate
OUTPUT_FRAME_DIR = './output_frames'
```

Thirdly, run the main script to start the tracking process:
```
 # full version 
python tractor_full.py

# without employing the appearance model
python tractor_no_appearance.py

# without employing the appearance model and the Kalman Filter
python tractor_no_appearance.py
```
The tracking results will be saved in video frames and video for visualization.


