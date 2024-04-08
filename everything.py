# encoding: utf-8
import os
import sys

import cv2

from me import generate_frames, VideoConfig, get_video_writer, Color, TextAnnotator, Detection, \
    filter_detections_by_class, detections2boxes, match_detections_with_tracks, BYTETrackerArgs, \
    get_player_in_possession, BaseAnnotator, BALL_COLOR, PLAYER_COLOR, REFEREE_COLOR, THICKNESS, MarkerAnntator, \
    BALL_MARKER_FILL_COLOR, PLAYER_MARKER_FILL_COLOR, GOALKEEPER_COLOR, COLORS

import torch

from yolox.tracker.byte_tracker import BYTETracker

from tqdm import tqdm

HOME = os.getcwd()
print('HOME: ', HOME)

WEIGHTS_PATH = f"{HOME}/data/best.pt"

SOURCE_VIDEO_PATH = f"{HOME}/clips/08fd33_4.mp4"
TARGET_VIDEO_PATH = f"{HOME}/final/8fd33_4.mp4"

frame_iterator = iter(generate_frames(video_file=SOURCE_VIDEO_PATH))
# print('frames length ', len(list(frame_iterator)))


model = torch.hub.load('ultralytics/yolov5', 'custom', WEIGHTS_PATH, device=0)
print(model.names)

# initiate video writer
video_config = VideoConfig(fps=30, width=1920, height=1080)

video_writer = get_video_writer(target_video_path=TARGET_VIDEO_PATH, video_config=video_config)

# initiate annotators
#text_annotator = TextAnnotator(background_color=Color(255, 255, 255), text_color=Color(0, 0, 0), text_thickness=2)

# initiate tracker
byte_tracker = BYTETracker(BYTETrackerArgs())

# distance in pixels from the player's bounding box where we consider the ball is in his possession
PLAYER_IN_POSSESSION_PROXIMITY = 30

# initiate annotators
base_annotator = BaseAnnotator(
    colors=COLORS,
    thickness=THICKNESS)

player_text_annotator = TextAnnotator(PLAYER_COLOR, text_color=Color(255, 255, 255), text_thickness=2, text_title="player")
goalkeeper_text_annotator = TextAnnotator(GOALKEEPER_COLOR, text_color=Color(255, 255, 255), text_thickness=2, text_title="keeper")
referee_text_annotator = TextAnnotator(REFEREE_COLOR, text_color=Color(0, 0, 0), text_thickness=2, text_title="referee")

# initiate annotators
ball_marker_annotator = MarkerAnntator(color=BALL_MARKER_FILL_COLOR)

player_marker_annotator = MarkerAnntator(color=PLAYER_MARKER_FILL_COLOR)

player_in_possession_marker_annotator = MarkerAnntator(color=PLAYER_MARKER_FILL_COLOR)

def main():
    #print('Python: {0}, {1}'.format(sys.platform, sys.version))
    # loop over frames
    for frame in tqdm(frame_iterator, total=750):
        # run detector
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

        # annotate video frame
        annotated_image = frame.copy()
        annotated_image = base_annotator.annotate(image=annotated_image, detections=tracked_detections)

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

        # save video frame
        video_writer.write(annotated_image)

        # close output video
    video_writer.release()

if __name__ == '__main__':
    main()
