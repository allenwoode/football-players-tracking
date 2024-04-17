# football-players-tracking

Football players tracking using YOLOV5+ByteTrack

## yolov5 usage

```shell
cd yolov5/
python detect.py --weights ../data/best.pt --img 1280 --conf 0.25 --source ../clips/08fd33_4.mp4 --name custom
```

## 
Detection and tracking is just the beginning. Now we can really take it to the next level! We can now quickly analyze 
the course of the action, knowing how the ball traveled between players, count the distance the players traveled, 
or locate the field zones where they appeared most often.
