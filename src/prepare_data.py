import pandas as pd
import joblib
import os
import glob
import json
import numpy as np
from tqdm import tqdm
# create the binarized labels for the categories that we will use.
from sklearn.preprocessing import LabelBinarizer 

DATA_PATH = '../hackathon_data_3000/'
REAL_DATA_PATH = '../hackathon_data_3000/REAL(1500)/'
SYN_DATA_PATH = '../hackathon_data_3000/SYN(1500)/'
REAL_DATA_VIDEO_PATH = '../hackathon_data_3000/REAL(1500)/0.REAL_VIDEO/'
REAL_DATA_WORD_META_PATH = '../hackathon_data_3000/REAL(1500)/REAL_WORD_morpheme/'
REAL_DATA_SENTENCE_META_PATH = '../hackathon_data_3000/REAL(1500)/REAL_SENTENCE_morpheme/'
TOP_N = 10

info = {} #{'NIA_SL_WORD0024_REAL02': [2.238, 4.063, '학교연혁'], '...'}
for file in glob.glob(REAL_DATA_PATH+'*_morpheme/'+'*.json'):
  with open(file) as json_file:
    data = json.load(json_file)
    video_name = data["metaData"]["name"] #NIA_SL_WORD0024_REAL02_D.mp4
    i = video_name.rfind('_')
    video_name_root = video_name[:i]  #NIA_SL_WORD0024_REAL02
    for morpheme in data["data"]:
      start = morpheme["start"] #2.238
      end = morpheme["end"] #4.063
      meaning = morpheme["attributes"][0]["name"] #학교연혁
      if not meaning in info:
        all_video_info = []
      else: 
        all_video_info = info[meaning]
      video_info = []
      video_info.append(video_name_root)
      video_info.append(start)
      video_info.append(end)
      all_video_info.append(video_info)
      info[meaning] = all_video_info
sorted_info = sorted(info.items(), key=lambda item: len(item[1]), reverse=True)
# for word in info[:10]:
#   print (word[0], len(word[1]))
top_sorted_info = sorted_info[:TOP_N]
top_sorted_dict = {}
for item in top_sorted_info:
  top_sorted_dict[item[0]] = item[1]
# print (top_sorted_dict)

with open('info.json', 'w', encoding='utf-8') as file:
  json.dump(top_sorted_dict, file, ensure_ascii=False)

# number <-> meaning mapping file
mapping = {}
count = 0
for key in top_sorted_dict.keys():
  mapping[key] = count
  count += 1
with open('mapping.json', 'w', encoding='utf-8') as file:
  json.dump(mapping, file, ensure_ascii=False)