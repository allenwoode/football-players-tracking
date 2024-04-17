# encoding: utf-8
import os
import sys

import cv2
import matplotlib.pyplot as plt

HOME = os.getcwd()
SOURCE_VIDEO_PATH = "./train/1606b0e6_0.mp4"
print(SOURCE_VIDEO_PATH)

def main():
    print('Python: {0}, {1}'.format(sys.platform, sys.version))
    vidcap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    success, img = vidcap.read()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(f"{SOURCE_VIDEO_PATH} fps: {fps}")
    frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(F"{SOURCE_VIDEO_PATH} frame: {frame_count}")
    plt.figure(figsize=(16, 8))
    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    main()
