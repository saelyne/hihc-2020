import json
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2

# learning_parameters 
lr = 1e-3
batch_size = 32
device = 'cuda:0'
print(f"Computation device: {device}\n")

with open('../input/data.json') as json_file:
  data_original = json.load(json_file) # {0: [[root_direction_path, length], [...], }

class VideoDataset(Dataset):
  def __init__(self, images_info, labels):
    self.X = images_info
    self.y = labels
  
  def __len__(self):
    return (len(self.X))
  
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
    label = np.tile(self.y[i], (file_count,1))
    ts_x = torch.tensor(video, dtype=torch.float)
    ts_y =  torch.tensor(label, dtype=torch.long)
    # print (ts_x.shape) #torch.Size([n, 3, 256, 256])
    # print (ts_y.shape) #torch.Size([n, 1])
    return (ts_x, ts_y)

data_new = []
for key, val in data_original.items():
  for video in val:
    item = video
    item.append(float(key))
    data_new.append(item)
data_frame = np.array(data_new)
X, y = np.split(data_frame, [-1], axis=1)
xtrain = X 
ytrain = y.astype(np.float)

train_data = VideoDataset(xtrain, ytrain)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
for i, data in enumerate(trainloader):
  a = 1