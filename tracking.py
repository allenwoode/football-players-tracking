# encoding: utf-8
import os
import sys

from me import generate_frames, VideoConfig, get_video_writer, Color, TextAnnotator, Detection, \
    filter_detections_by_class, detections2boxes, match_detections_with_tracks, BYTETrackerArgs

import torch

from yolox.tracker.byte_tracker import BYTETracker

from tqdm.notebook import tqdm

HOME = os.getcwd()
print('HOME: ', HOME)

WEIGHTS_PATH = f"{HOME}/data/best.pt"

SOURCE_VIDEO_PATH = f"{HOME}/clips/08fd33_4.mp4"
TARGET_VIDEO_PATH = f"{HOME}/tracking/8fd33_4.mp4"

frame_iterator = iter(generate_frames(video_file=SOURCE_VIDEO_PATH))

model = torch.hub.load('ultralytics/yolov5', 'custom', WEIGHTS_PATH, device=0)

# initiate video writer
video_config = VideoConfig(fps=30, width=1920, height=1080)
print(video_config)

video_writer = get_video_writer(target_video_path=TARGET_VIDEO_PATH, video_config=video_config)

# initiate annotators
text_annotator = TextAnnotator(background_color=Color(255, 255, 255), text_color=Color(0, 0, 0), text_thickness=2)

# initiate tracker
byte_tracker = BYTETracker(BYTETrackerArgs())

def main():
    #print('Python: {0}, {1}'.format(sys.platform, sys.version))
    # loop over frames
    for frame in tqdm(frame_iterator, total=750):
        # run detector
        results = model(frame, size=1280)
        detections = Detection.from_results(
            pred=results.pred[0].cpu().numpy(),
            names=model.names)

        # postprocess results
        goalkeeper_detections = filter_detections_by_class(detections=detections, class_name="goalkeeper")
        player_detections = filter_detections_by_class(detections=detections, class_name="player")
        player_detections = player_detections + goalkeeper_detections

        # track players
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=player_detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        player_detections = match_detections_with_tracks(detections=player_detections, tracks=tracks)

        # annotate video frame
        annotated_image = frame.copy()
        annotated_image = text_annotator.annotate(
            image=annotated_image,
            detections=player_detections)

        # save video frame
        video_writer.write(annotated_image)

    # close output video
    video_writer.release()

if __name__ == '__main__':
    main()
