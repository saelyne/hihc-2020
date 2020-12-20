import cv2
import glob
import json
import os
import numpy as np

with open('info.json', encoding='utf-8') as json_file:
  info = json.load(json_file)

with open('mapping.json', encoding='utf-8') as json_file:
  mapping = json.load(json_file)

REAL_DATA_PATH = '../hackathon_data_3000/REAL(1500)/0.REAL_VIDEO'
SAVE_DATA_PATH = '../input/data_np'
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
    
    ret, frame = video.read()
    image = cv2.resize(frame, (256, 256), interpolation = cv2.INTER_CUBIC)
    image = np.asarray(image, dtype="int32") #img.shape: (256, 256, 3)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32) #img.shape: (3, 256, 256)
    frames = np.array([image])
    print(frames.shape)
    input()
    labels = [10]

    counter = 1
    while (video.isOpened()):
      ret, frame = video.read()
      if ret == True:
        image = cv2.resize(frame, (256, 256), interpolation = cv2.INTER_CUBIC)
        image = np.asarray(image, dtype="int32") #img.shape: (256, 256, 3)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) #img.shape: (3, 256, 256)
        image = np.array([image])
        frames = np.concatenate((frames, image), 0)
        if start <= counter/30.0 and counter/30.0 <=end:
          labels.append(meaning)
        else:
          labels.append(10)
      else:
        break
      counter += 1

    labels = np.array([labels])
    print(labels.shape)
    input()
    X_path = f"{SAVE_DATA_PATH}/{meaning}/{video_root}/{video_root}_{direction}/x.npy"
    Y_path = f"{SAVE_DATA_PATH}/{meaning}/{video_root}/{video_root}_{direction}/y.npy"

    video.release()
  # cv2.destroyAllWindows()

count = 0
for key, val in info.items():
  meaning = mapping[key]
  video_list = val
  print ("new meaning")
  for val in video_list:
    print ("new video")
    video_root = val[0]
    save_frame(meaning, video_root, val[1], val[2])
  count += 1

# Should we include below?
# cv2.destroyAllWindows()