# encoding: utf-8
import os
import sys

import torch

from tqdm import tqdm

from me import generate_frames, plot_image

HOME = os.getcwd()
print('HOME: ', HOME)

WEIGHTS_PATH = f"{HOME}/data/best.pt"
SOURCE_VIDEO_PATH = f"{HOME}/clips/08fd33_4.mp4"
frame_iterator = iter(generate_frames(video_file=SOURCE_VIDEO_PATH))

model = torch.hub.load('ultralytics/yolov5', 'custom', WEIGHTS_PATH, device=0)
print(model.names)

def main():
    print('Python: {0}, {1}'.format(sys.platform, sys.version))
    # initiate annotators
    #annotator = BaseAnnotator(colors=COLORS, thickness=THICKNESS)

    frames = []
    # acquire video frame
    for frame in tqdm(frame_iterator, total=50):
        frames.append(frame)

    print(len(frames))
    # run detector
    # results = model(frame, size=1280)
    # detections = Detection.from_results(pred=results.pred[0].cpu().numpy(), names=model.names)
    #
    # # annotate video frame
    # annotated_image = annotator.annotate(image=frame, detections=detections)
    #
    # # plot video frame
    # plot_image(annotated_image, 16)




if __name__ == '__main__':
    main()
