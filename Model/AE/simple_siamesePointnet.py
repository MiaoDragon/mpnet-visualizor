"""
The original pointnet implementation is from
ref: https://github.com/fxia22/pointnet.pytorch
Siamese features is added upon that

Adding Siamese feature notice:
1d conv layers weights are already shared
fc layers need to appy separately, and then apply bn together
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from model.simple_pointnet import *

class SiamesePointNet(nn.Module):
    def __init__(self, feature_transform=False):
        super(SiamesePointNet, self).__init__()
        self.feature_transform=feature_transform
        # concatenate golbal features to local features
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(320, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 64, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        # concatenate local features into global features
        # then use fully connected layer to obtain global features
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)
        self.dropout = nn.Dropout(p=0.3)
        self.fc_bn1 = nn.BatchNorm1d(64)
        self.fc_bn2 = nn.BatchNorm1d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [B*2, shape]
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        # concatenate golbal features to local features
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # concatenate local features into global features
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 64)
        # then use fully connected layer to obtain global features
        x = F.relu(self.fc_bn1(self.fc1(x)))
        # below is inspired by
        # https://becominghuman.ai/siamese-networks-algorithm-applications-and-pytorch-implementation-4ffa3304c18
        x = self.fc2(x)
        div_pos = len(x) // 2
        x = torch.abs(x[:div_pos] - x[div_pos:])
        x = self.fc3(x)  # used for classification (either similar or not)
        return x, trans, trans_feat
