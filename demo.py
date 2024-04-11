import os
import sys

import cv2
import torch

from PIL import Image

HOME = os.getcwd()
print(HOME)

WEIGHT_PATH = f"{HOME}/yolov5/yolov5x.pt"
#
# # Model
#model = torch.hub.load('ultralytics/yolov5', 'coco', path=WEIGHT_PATH)
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='data/best.pt', )  # local model
#model = torch.hub.load('./yolov5', 'custom', path='./yolov5/yolov5x.pt', source='local')  # local repo
#model = torch.hub.load('ultralytics/yolov5', 'coco', path='./yolov5/yolov5x.pt')  # PyTorch
#
# # Image
im = 'https://ultralytics.com/images/zidane.jpg'

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Images
for f in 'zidane.jpg', 'bus.jpg':
    torch.hub.download_url_to_file('https://ultralytics.com/images/' + f, f)  # download 2 images
im1 = Image.open('zidane.jpg')  # PIL image
im2 = cv2.imread('bus.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)

# Inference
results = model([im1, im2], size=640)  # batch of images

# Results
results.print()
results.save()  # or .show()

#results = model(im)

def main():
    print("{}: {}".format(sys.platform, sys.version))

    #print(results.pandas().xyxy[0])
    #print(torch.hub.help('pytorch/vision', 'resnet18', force_reload=True))


if __name__ == "__main__":
    main()
