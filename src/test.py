'''
USAGE:
python test.py --model ../output/hand.pth
# --input ../input/example_clips/NIA_SL_SEN0001_REAL10_D.mp4
'''
import torch
import numpy as np
import argparse
import joblib
import cv2
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision.transforms import transforms   
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from msm import MultiStageModel

TEST_PATH = '../input/example_clips/NIA_SL_SEN0047_REAL01_F.mp4'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# construct the argument parser
"""
--model is the path to the saved model on the disk.
--label-bin gives the path to the saved binarized labels files. We have saved this file while executing the prepare_data.py file.
--input is the path to the input video clips that we will test our model on.
--outputs is the path to save the output video clips after the video recognition takes place.
"""
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True,
    help="path to trained serialized model")
# ap.add_argument('-i', '--input', required=True,
#     help='path to our input video')
# ap.add_argument('-l', '--label-bin', required=True,
#     help="path to  label binarizer")
# ap.add_argument('-o', '--output', required=True, type=str,
#     help='path to our output video')
args = vars(ap.parse_args())

# load the trained model and label binarizer from disk
# print('Loading model and label binarizer...')
# lb = joblib.load(args['label_bin'])
model = MultiStageModel(4, 10, 64, 3364, 11).cuda()
print('Model Loaded...')
model.load_state_dict(torch.load(args['model']))
print('Loaded model state_dict...')
# aug = albumentations.Compose([
#     albumentations.Resize(224, 224),
#     ])

cap = cv2.VideoCapture(TEST_PATH)
if (cap.isOpened() == False):
    print('Error while trying to read video. Plese check again...')
# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# define codec and create VideoWriter object
# out = cv2.VideoWriter(str(args['output']), cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))

# read until end of video
ret, frame = cap.read()
with torch.no_grad():
    image = cv2.resize(frame, (256, 256), interpolation = cv2.INTER_CUBIC)
    image = np.asarray(image, dtype="int32") #img.shape: (256, 256, 3)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32) #img.shape: (3, 256, 256)
    video = torch.from_numpy(image).unsqueeze(0)

model.eval()
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        # model.eval()
        with torch.no_grad():
            image = cv2.resize(frame, (256, 256), interpolation = cv2.INTER_CUBIC)
            image = np.asarray(image, dtype="int32") #img.shape: (256, 256, 3)
            image = np.transpose(image, (2, 0, 1)).astype(np.float32) #img.shape: (3, 256, 256)
            image = torch.from_numpy(image).unsqueeze(0)
            video = torch.cat((video, image), 0)
        
        # cv2.putText(frame, lb.classes_[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
        # cv2.imshow('image', frame)
        # out.write(frame)
        # press `q` to exit
        # if cv2.waitKey(27) & 0xFF == ord('q'):
        #     break
    else: 
        break

video = video.to(device)
Y_hat, predictions = model(video)
_, preds = torch.max(Y_hat.data, 2)
print (preds)

# release VideoCapture()
cap.release()
# close all frames and video windows
# cv2.destroyAllWindows()