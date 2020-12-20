# save data into json file
import os
import json

DATA_PATH = '../input/data'
# create a DataFrame
data = {}

for class_folder in os.listdir(f"{DATA_PATH}"):
  if class_folder.startswith('.'):
    continue
  word_list = []
  for video_folder in os.listdir(f"{DATA_PATH}/{class_folder}"):
    for direction_folder in os.listdir(f"{DATA_PATH}/{class_folder}/{video_folder}"):
      folder = f"{DATA_PATH}/{class_folder}/{video_folder}/{direction_folder}/"
      img_list = [folder, len(os.listdir(folder))]
      word_list.append(img_list)
  data[class_folder] = word_list


# save as json file
with open('../input/data.json', 'w') as file:
  json.dump(data, file)