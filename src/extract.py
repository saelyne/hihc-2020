import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split

from pytorch_i3d import InceptionI3d
# learning_parameters 
lr = 1e-3
batch_size = 1
num_classes = 10
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computation device: {device}\n")

class VideoDataset(Dataset):
  def __init__(self, images_info, labels):
    self.X = images_info
    self.y = labels
  
  def __len__(self):
    return len(self.X)
  
  def __getitem__(self, i):
    info = self.X[i]
    file_path_root = info[0]
    file_count = int(info[1])
    i = 0 
    video = torch.zeros(file_count, 3, 256, 256).to(device)
    for i in range(file_count):
      img_path = file_path_root + str(i) + '.jpg'
      image = cv2.imread(img_path)
      image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_CUBIC)
      image = np.asarray(image, dtype="int32" ) #img.shape: (256, 256, 3)
      image = np.transpose(image, (2, 0, 1)).astype(np.float32) #img.shape: (3, 256, 256)
      video[i] = torch.from_numpy(image)
    label = np.tile(self.y[i], (file_count))
    ts_x = video.float()
    ts_y =  torch.LongTensor(label)
    
    return ts_x, ts_y, file_path_root[14:-1]

with open('../input/data.json') as json_file:
  data_original = json.load(json_file) # {0: [[root_direction_path, length], [...], }
  
data_new = []
for key, val in data_original.items():
  for video in val:
    item = video
    item.append(float(key))
    data_new.append(item)
data_frame = np.array(data_new)

X, y = np.split(data_frame, [-1], axis=1)
Y = y.astype(np.float)

new_X = []
new_Y = []
for i, x in enumerate(X):
    if int(x[1]) >= 10:
        new_X.append(x)
        new_Y.append(Y[i])

new_X = np.array(new_X)
new_Y = np.array(new_Y)

train_data = VideoDataset(new_X, new_Y)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


load_model = "../input/rgb_charades.pt"
save_dir = "../input/i3d"

i3d = InceptionI3d(400, in_channels=3)
i3d.replace_logits(157)
i3d.load_state_dict(torch.load(load_model))
i3d.cuda()

i3d.train(False)  # Set model to evaluate mode
        
tot_loss = 0.0
tot_loc_loss = 0.0
tot_cls_loss = 0.0
            
# Iterate over data.
for data in trainloader:
    # get the inputs
    inputs, labels, name = data
    if os.path.exists(os.path.join(save_dir, name[0]+'.npy')):
        continue

    wor, vid, fil = name[0].split('/')
    
    if not os.path.exists(os.path.join(save_dir,wor)):
        os.mkdir(os.path.join(save_dir,wor))
    if not os.path.exists(os.path.join(save_dir,wor,vid)):
        os.mkdir(os.path.join(save_dir,wor,vid))

    inputs = inputs.transpose(2, 1)
    b,c,t,h,w = inputs.shape
    print(name[0], inputs.shape)
    if t > 1600:
        features = []
        for start in range(1, t-56, 1600):
            end = min(t-1, start+1600+56)
            start = max(1, start-48)
            ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]).cuda(), volatile=True)
            features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
        np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
    else:
        # wrap them in Variable
        inputs = Variable(inputs.cuda(), volatile=True)
        features = i3d.extract_features(inputs)
        a = features.squeeze(0).permute(1,2,3,0).data.cpu().numpy()
        print(a.shape)
        exit()
        np.save(os.path.join(save_dir, name[0]), features.squeeze(0).permute(1,2,3,0).data.cpu().numpy())
