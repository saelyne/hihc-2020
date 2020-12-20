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

class MultiStageModel(nn.Module):
  def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
    super(MultiStageModel, self).__init__()
    self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
    self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])
    self.to_feature_1 = nn.Conv2d(3, 16, 16, 4)
    self.to_feature_2 = nn.Conv2d(16, 1, 4, 1)

  def forward(self, x):
    # x.shape = [1, 16, 3, 256, 256]

    x = self.to_feature_1(x.squeeze())
    x = self.to_feature_2(x)
    x = x.reshape(x.shape[0], -1)
    x = x.unsqueeze(0)
    x = x.transpose(2, 1)

    # x.shape = (1, 2048, 6046)
    out = self.stage1(x)
    outputs = out.unsqueeze(0)
    for s in self.stages:
      out = s(F.softmax(out, dim=1))
      outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
    # outputs[-1].shape = (1, 19, 6046)
    return outputs[-1].transpose(2, 1), outputs


class SingleStageModel(nn.Module):
  def __init__(self, num_layers, num_f_maps, dim, num_classes):
    super(SingleStageModel, self).__init__()
    self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
    self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
    self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

  def forward(self, x):
    # x.shape = (1, 2048, 6046)
    out = self.conv_1x1(x)
    # out.shape = (1, 64, 6046)
    for layer in self.layers:
      out = layer(out)
    out = self.conv_out(out)
    return out


class DilatedResidualLayer(nn.Module):
  def __init__(self, dilation, in_channels, out_channels):
    super(DilatedResidualLayer, self).__init__()
    self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
    self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
    self.dropout = nn.Dropout()

  def forward(self, x):
    # x.shape = (1, 64, 6046)
    out = F.relu(self.conv_dilated(x))
    out = self.conv_1x1(out)
    out = self.dropout(out)
    return (x + out)