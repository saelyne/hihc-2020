import cv2
import glob
import json
import os
from PIL import Image
import numpy as np

with open('info.json', encoding='utf-8') as json_file:
  info = json.load(json_file)

REAL_DATA_PATH = '../hackathon_data_3000/REAL(1500)/0.REAL_VIDEO'
SAVE_DATA_PATH = '../input/data'
# ROOT_PATH = '/Users/Saelyne/Documents/Activity/2020-HIHC/silent-speaker'
DIRECTIONS = ['F', 'U', 'D', 'R', 'L']

def save_frame(meaning, video_root, start, end):
  for direction in DIRECTIONS:
    video_path = f"{REAL_DATA_PATH}/{video_root}_{direction}.mp4"
    video = cv2.VideoCapture(video_path)
    counter = 0
    index = 0

    meaning_path = f"{SAVE_DATA_PATH}/{meaning}"
    if not os.path.isdir(meaning_path):
      os.mkdir(meaning_path)
    video_root_path = f"{SAVE_DATA_PATH}/{meaning}/{video_root}"
    if not os.path.isdir(video_root_path):
      os.mkdir(video_root_path)
    direction_path = f"{SAVE_DATA_PATH}/{meaning}/{video_root}/{video_root}_{direction}"
    if not os.path.isdir(direction_path):
      os.mkdir(direction_path)

    #frame rate = 30fps (30 frame / second)
    while (video.isOpened()):
      ret, frame = video.read()
      if ret == False:
        break
      if start <= counter/30.0 and counter/30.0 <=end:
        img_path = f"{SAVE_DATA_PATH}/{meaning}/{video_root}/{video_root}_{direction}/{index}.jpg"
        frame = cv2.resize(frame, (256, 256), interpolation = cv2.INTER_CUBIC)
        data = np.asarray( frame, dtype="int32" )
        print (data.shape)
      counter+=1
    video.release()
  # cv2.destroyAllWindows()

count = 0
for items in info.items():
  video_list = items[1]
  print ("new meaning")
  for val in video_list:
    print ("new video")
    video_root = val[0]
    save_frame(count, video_root, val[1], val[2])
  count += 1

# Should we include below?
# cv2.destroyAllWindows()