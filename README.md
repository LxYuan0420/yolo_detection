#### Real time/images/video object detection using YOLO pretrained model
---



1. #### Getting started:
```zsh
$ git clone https://github.com/LxYuan0420/yolo_detection.git
$ cd yolo_detection

$ cd models/yolov3/
$ wget https://pjreddie.com/media/files/yolov3-tiny.weights
$ cd -

$ python3 -m venv env
$ source env/bin/activate
(env)$ pip install -r requirements.txt

```

2. #### Run inference on images
```zsh
(env)$ python yolo.py --image data/images/malaysia_traffic_police.jpg --model models/yolov3

[INFO] loading YOLO from disk...
[INFO] YOLO took 0.073651 seconds
```


![Screenshot 2022-07-09 at 3 54 22 PM](https://user-images.githubusercontent.com/25739658/178097166-927245d8-4e01-4773-b3dd-fd959adea1f9.png)

3. #### Run inference on videos
```zsh
(env)$ pip install youtube-dl
(env)$ cd data/videos/
(env)$ youtube-dl https://youtu.be/FgV8j_Syww4 
(env)$ cd -
(env)$ python yolo_video.py \
--input data/videos/stephen_currey_basketball.mp4 \
--output data/video_outputs/stephen_currey_basketball.avi \
--model models/yolov3

[INFO] loading YOLO from disk...
[INFO] 2138 total frames in video
[INFO] single frame took 0.0600 seconds
[INFO] estimated total time to finish: 128.3574
[INFO] cleaning up...

(env)$ open data/video_outputs/stephen_currey_basketball.avi
```
![Screenshot 2022-07-09 at 4 10 53 PM](https://user-images.githubusercontent.com/25739658/178097588-275613c4-361f-4111-b286-14bb5133c6e9.png)


4. #### Run inference on webcam
```zsh
(env)$ python yolo_webcam.py
```
![Screenshot 2022-07-09 at 3 25 35 PM](https://user-images.githubusercontent.com/25739658/178097503-81d46627-19e5-4ae4-8689-e2ebe4eb43f8.png)

5. #### [OPTIONAL] Setup pre-commit hook
```zsh
(env)$ pip install -r requirements_dev.txt
(env)$ pre-commit install
(env)$ pre-commit run --all-files
```
#### Repo structure:
```
.
├── data
│   ├── images
│   │   └── malaysia_traffic_police.jpg
│   ├── video_outputs
│   │   └── stephen_currey_basketball.avi
│   └── videos
│       └── stephen_currey_basketball.mp4
├── env
├── models
│   └── yolov3
│       ├── coco.names
│       ├── yolov3-tiny.cfg
│       └── yolov3-tiny.weights
├── requirements.txt
├── requirements_dev.txt
├── setup.cfg
├── yolo.py
├── yolo_video.py
└── yolo_webcam.py
```

##### Reference:
- [Darknet](https://pjreddie.com/darknet/yolo/)
- [YOLO object detection with OpenCV](https://pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
- [Implementation of YOLOv3: Simplified](https://www.analyticsvidhya.com/blog/2021/06/implementation-of-yolov3-simplified/)

