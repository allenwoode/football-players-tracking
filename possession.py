# encoding: utf-8
import os
import sys

import cv2
import torch

from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation

import me
from me import generate_frames, MarkerAnntator, BALL_MARKER_FILL_COLOR, PLAYER_MARKER_FILL_COLOR, Detection, \
    filter_detections_by_class, VideoConfig, get_video_writer, get_player_in_possession


HOME = os.getcwd()
# settings
SOURCE_VIDEO_PATH = f"{HOME}/clips/08fd33_4.mp4"
TARGET_VIDEO_PATH = f"{HOME}/ball-possession/8fd33_4.mp4"

WEIGHTS_PATH = f"{HOME}/data/best.pt"

# initiate video writer
video_config = VideoConfig(fps=30, width=1920, height=1080)
print(video_config)

video_writer = get_video_writer(target_video_path=TARGET_VIDEO_PATH, video_config=video_config)

# get fresh video frame generator
frame_iterator = iter(generate_frames(video_file=SOURCE_VIDEO_PATH))

# initiate annotators
ball_marker_annotator = MarkerAnntator(color=BALL_MARKER_FILL_COLOR)
player_marker_annotator = MarkerAnntator(color=PLAYER_MARKER_FILL_COLOR)

model = torch.hub.load('ultralytics/yolov5', 'custom', WEIGHTS_PATH, device=0)
#print(model.names)

# distance in pixels from the player's bounding box where we consider the ball is in his possession
PLAYER_IN_POSSESSION_PROXIMITY = 30

def snapshot(frame):
    results = model(frame, size=1280)

    detections = Detection.from_results(
        pred=results.pred[0].cpu().numpy(),
        names=model.names)

    # post process results
    ball_detections = filter_detections_by_class(detections=detections, class_name="ball")
    goalkeeper_detections = filter_detections_by_class(detections=detections, class_name="goalkeeper")
    player_detections = filter_detections_by_class(detections=detections, class_name="player") + goalkeeper_detections

    player_in_possession_detection = get_player_in_possession(
        player_detections=player_detections,
        ball_detections=ball_detections,
        proximity=PLAYER_IN_POSSESSION_PROXIMITY)

    # annotate video frame
    annotated_image = frame.copy()
    annotated_image = ball_marker_annotator.annotate(
        image=annotated_image,
        detections=ball_detections)
    annotated_image = player_marker_annotator.annotate(
        image=annotated_image,
        detections=[player_in_possession_detection] if player_in_possession_detection else [])

    return annotated_image


def main():
    #print('Python: {0}, {1}'.format(sys.platform, sys.version))
    # loop over frames
    for i, frame in enumerate(frame_iterator):
        # run detector
        annotated_image = snapshot(frame)
        print(f'video writer frame {i}')
        # save video frame
        video_writer.write(annotated_image)

    # close output video
    video_writer.release()

def photo():
    frame = next(frame_iterator)
    annotated_image = snapshot(frame)
    cv2.imwrite(f"{HOME}/sample/possession.jpg", annotated_image)

def plot():
    frame = next(frame_iterator)
    annotated_image = snapshot(frame)
    me.plot_image(annotated_image, 16)

def animate():
    fig, ax = plt.subplots()

    fps = 30
    frames = list(frame_iterator)
    frame = frames[0]
    im = plt.imshow(frame, interpolation='none', aspect='auto', vmin=0, vmax=1)

    def update(i):
        # for each frame, update the data stored on each artist.
        # if i % fps == 0:
        #     print('.', end='')
        out = snapshot(frames[i])

        im.set_array(out)
        return [im]

    ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=1000 / fps)
    # ani.save(f'{HOME}/video/animation.mp4')
    plt.show()

if __name__ == '__main__':
    #main()
    animate()
    print('- done -')
    # for i, frame in enumerate(frame_iterator):
    #     print(i, frame.shape)