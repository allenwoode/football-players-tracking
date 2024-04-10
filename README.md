# football-players-tracking

Football players tracking using YOLOV5+ByteTrack

## yolov5 usage

```shell
cd yolov5/
python detect.py --weights ../data/best.pt --img 1280 --conf 0.25 --source ../clips/08fd33_4.mp4 --name custom
```