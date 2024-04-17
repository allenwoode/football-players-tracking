# encoding: utf-8
import os
import sys

import cv2
import torch

from me import generate_frames, BaseAnnotator, COLORS, Detection, filter_detections_by_class, \
    BYTETrackerArgs, detections2boxes, match_detections_with_tracks, TextAnnotator, PLAYER_COLOR, GOALKEEPER_COLOR, \
    REFEREE_COLOR, Color, MarkerAnnotator, BALL_MARKER_FILL_COLOR, get_player_in_possession, PLAYER_MARKER_FILL_COLOR
from yolox.tracker.byte_tracker import BYTETracker

HOME = os.getcwd()
print('HOME: ', HOME)

WEIGHTS_PATH = f"{HOME}/data/best.pt"
SOURCE_VIDEO_PATH = f"{HOME}/clips/08fd33_1.mp4"
frame_iterator = iter(generate_frames(video_file=SOURCE_VIDEO_PATH))

model = torch.hub.load('ultralytics/yolov5', 'custom', WEIGHTS_PATH, device=0)
print(model.names)

PLAYER_IN_POSSESSION_PROXIMITY = 30

# initiate tracker
byte_tracker = BYTETracker(BYTETrackerArgs())

thickness = 2 #字体大小

# initiate annotators
base_annotator = BaseAnnotator(colors=COLORS, thickness=4)

player_text_annotator = TextAnnotator(PLAYER_COLOR, text_color=Color(255, 255, 255), text_thickness=thickness)
goalkeeper_text_annotator = TextAnnotator(GOALKEEPER_COLOR, text_color=Color(255, 255, 255), text_thickness=thickness)
referee_text_annotator = TextAnnotator(REFEREE_COLOR, text_color=Color(0, 0, 0), text_thickness=thickness)

# initiate annotators
ball_marker_annotator = MarkerAnnotator(color=BALL_MARKER_FILL_COLOR)
player_marker_annotator = MarkerAnnotator(color=PLAYER_MARKER_FILL_COLOR)
#player_in_possession_marker_annotator = MarkerAnntator(color=PLAYER_MARKER_FILL_COLOR)

def snapshot(frame):
    results = model(frame, size=1280)
    detections = Detection.from_results(
        pred=results.pred[0].cpu().numpy(),
        names=model.names)

    # filter detections by class
    ball_detections = filter_detections_by_class(detections=detections, class_name="ball")
    player_detections = filter_detections_by_class(detections=detections, class_name="player")
    goalkeeper_detections = filter_detections_by_class(detections=detections, class_name="goalkeeper")
    referee_detections = filter_detections_by_class(detections=detections, class_name="referee")

    tracked_detections = player_detections + goalkeeper_detections + referee_detections

    player_goalkeeper_detections = player_detections + goalkeeper_detections
    # calculate player in possession
    player_in_possession_detection = get_player_in_possession(
        player_detections=player_goalkeeper_detections,
        ball_detections=ball_detections,
        proximity=PLAYER_IN_POSSESSION_PROXIMITY)

    # track
    tracks = byte_tracker.update(
        output_results=detections2boxes(detections=tracked_detections),
        img_info=frame.shape,
        img_size=frame.shape
    )
    tracked_detections = match_detections_with_tracks(detections=tracked_detections, tracks=tracks)

    tracked_referee_detections = filter_detections_by_class(detections=tracked_detections, class_name="referee")
    tracked_goalkeeper_detections = filter_detections_by_class(detections=tracked_detections, class_name="goalkeeper")
    tracked_player_detections = filter_detections_by_class(detections=tracked_detections, class_name="player")

    # annotate video frame 标注视频帧
    annotated_image = frame.copy()
    annotated_image = base_annotator.annotate(image=annotated_image, detections=detections)

    annotated_image = player_text_annotator.annotate(
        image=annotated_image,
        detections=tracked_player_detections)
    annotated_image = goalkeeper_text_annotator.annotate(
        image=annotated_image,
        detections=tracked_goalkeeper_detections)
    annotated_image = referee_text_annotator.annotate(
        image=annotated_image,
        detections=tracked_referee_detections)

    annotated_image = ball_marker_annotator.annotate(
        image=annotated_image,
        detections=ball_detections)
    annotated_image = player_marker_annotator.annotate(
        image=annotated_image,
        detections=[player_in_possession_detection] if player_in_possession_detection else [])

    return annotated_image


def photo(i):
    frame = list(frame_iterator)[i]
    annotated_image = snapshot(frame)
    cv2.imwrite(f"{HOME}/final/snapshot-{i}.jpg", annotated_image)

def main():
    print('Python: {0}, {1}'.format(sys.platform, sys.version))
    # acquire video frame
    photo(740)
    print('-- done --')


if __name__ == '__main__':
    main()
