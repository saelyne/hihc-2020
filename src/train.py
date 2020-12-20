'''
USAGE:
python train.py --model ../output/hand.pth
'''
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
import argparse
import matplotlib
import matplotlib.pyplot as plt
from msm import MultiStageModel

# learning_parameters 
lr = 1e-3
batch_size = 1
num_classes = 10
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computation device: {device}\n")

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
# ap.add_argument('-l', '--label-bin', required=True,
#     help="path to  label binarizer")
# ap.add_argument('-i', '--input', required=True,
#     help='path to our input video')
# ap.add_argument('-o', '--output', required=True, type=str,
#     help='path to our output video')
args = vars(ap.parse_args())

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
    j = 0 
    video = torch.zeros(file_count, 3, 256, 256)
    for j in range(file_count):
      img_path = file_path_root + str(j) + '.jpg'
      image = cv2.imread(img_path)
      image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_CUBIC)
      image = np.asarray(image, dtype="int32" ) #img.shape: (256, 256, 3)
      image = np.transpose(image, (2, 0, 1)).astype(np.float32) #img.shape: (3, 256, 256)
      video[j] = torch.from_numpy(image)
    label = np.tile(self.y[i], (file_count))
    ts_x_1 = video.float()
    ts_y_1 =  torch.LongTensor(label)
    # print (ts_x.shape) #torch.Size([n, 3, 256, 256])
    # print (ts_y.shape) #torch.Size([n, 1])
    
    i = random.randint(0,len(self.X)-1) 
    info = self.X[i]
    file_path_root = info[0]
    file_count = int(info[1])
    j = 0 
    video = torch.zeros(file_count, 3, 256, 256)
    for j in range(file_count):
      img_path = file_path_root + str(j) + '.jpg'
      image = cv2.imread(img_path)
      image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_CUBIC)
      image = np.asarray(image, dtype="int32") #img.shape: (256, 256, 3)
      image = np.transpose(image, (2, 0, 1)).astype(np.float32) #img.shape: (3, 256, 256)
      video[j] = torch.from_numpy(image)
    label = np.tile(self.y[i], (file_count))
    ts_x_2 = video.float()
    ts_y_2 =  torch.LongTensor(label)
    # ts_x = torch.cat((ts_x, torch.tensor(video, dtype=torch.float)), 0)
    # ts_y =  torch.cat((ts_y, torch.tensor(label, dtype=torch.long)), 0)
    ts_x = torch.cat((ts_x_1, ts_x_2), 0).to(device)
    ts_y = torch.cat((ts_y_1, ts_y_2), 0).to(device)

    return (ts_x, ts_y)

with open('../input/data.json') as json_file:
  data_original = json.load(json_file) # {0: [[root_direction_path, length], [...], }
  
data_new = []
for key, val in data_original.items():
  for video in val:
    item = video
    item.append(key)
    data_new.append(item)
data_frame = np.array(data_new)

X, y = np.split(data_frame, [-1], axis=1)
Y = y.astype(np.float)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
train_data = VideoDataset(X_train, y_train)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = VideoDataset(X_test, y_test)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = MultiStageModel(4, 10, 64, 3364, num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
ce = nn.CrossEntropyLoss(ignore_index=-100)
mse = nn.MSELoss(reduction='none')
def tmse(Y):
  return torch.mean(torch.clamp(mse(F.log_softmax(Y[:, :, 1:], dim=1), F.log_softmax(Y.detach()[:, :, :-1], dim=1)), min=0, max=16))

N = len(trainloader)

train_loss , train_accuracy = [], []
val_loss , val_accuracy = [], []

for epoch in range(num_epochs):
  epoch_loss = 0
  epoch_correct = 0
  epoch_total = 0
  epoch_correct_count = 0
  for i, data in enumerate(trainloader):
    # if i % 20 == 0:
    #   print(f"{i}/{len(trainloader)}")
    x, y = data
    optimizer.zero_grad()

    # x.shape = [1, 15, 3, 256, 256]
    Y_hat, predictions = model(x)
    # predictions = (4, 1, 19, 6046)
    # y = (1, 6046)
    loss = 0
    for p in predictions:
      loss += ce(p.transpose(2, 1).contiguous().view(-1, num_classes), y.view(-1)) + 0.15 * tmse(p)
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(Y_hat.data, 2)
    correct = ((predicted == y).float()).sum().item()
    total = Y_hat.shape[1]

    # print(f"{i}/{N}. loss: {loss.item()}, correct frames: {correct}/{total}={float(correct)/total} {y[0][0]}")

    epoch_loss += loss.item()
    epoch_correct += correct
    epoch_total += total
    if float(correct) / total > 0.8:
      epoch_correct_count += 1
  accuracy = float(epoch_correct)/epoch_total
    # print(y.shape) [1, 34]
    # print(Y_hat.shape) [1, 34, 10]
    # print(predictions.shape) [4, 1, 10, 34]
    # exit()
  train_loss.append(epoch_loss)
  train_accuracy.append(accuracy)
  print(f"Train\t epoch_loss: {epoch_loss}, accuracy: {accuracy}, count: {epoch_correct_count}, epoch: {epoch}")
  
  epoch_correct = 0
  epoch_total = 0
  epoch_correct_count = 0
  for i, data in enumerate(testloader):
    # if i % 20 == 0:
    #   print(f"{i}/{len(testloader)}")
    x, y = data
    Y_hat, predictions = model(x)
    _, predicted = torch.max(Y_hat.data, 2)
    correct = ((predicted == y).float()).sum().item()
    total = Y_hat.shape[1]

    epoch_correct += correct
    epoch_total += total
    if float(correct) / total > 0.8:
      epoch_correct_count += 1
  accuracy = float(epoch_correct)/epoch_total
  val_loss.append(epoch_loss)
  val_accuracy.append(accuracy)
  print(f"Test \t epoch_loss: {epoch_loss}, accuracy: {accuracy}, count: {epoch_correct_count}")

# accuracy plots
plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color='green', label='train accuracy')
plt.plot(val_accuracy, color='blue', label='validataion accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('../output/accuracy.png')
plt.show()
# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../output/loss.png')
plt.show()

# serialize the model to disk
print('Saving model...')
torch.save(model.state_dict(), args['model'])
 
print('TRAINING COMPLETE')