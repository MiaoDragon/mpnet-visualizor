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
from model.pointnet import *

class SiamesePointNet(nn.Module):
    def __init__(self, feature_transform=False):
        super(SiamesePointNet, self).__init__()
        self.feature_transform=feature_transform
        # concatenate golbal features to local features
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        # concatenate local features into global features
        # then use fully connected layer to obtain global features
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(p=0.3)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc_bn2 = nn.BatchNorm1d(256)
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
        x = x.view(-1, 256)
        # then use fully connected layer to obtain global features
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = F.relu(self.fc_bn2(self.dropout(self.fc2(x))))
        # below is inspired by
        # https://becominghuman.ai/siamese-networks-algorithm-applications-and-pytorch-implementation-4ffa3304c18
        x = self.fc3(x)
        div_pos = len(x) // 2
        x = torch.abs(x[:div_pos] - x[div_pos:])
        x = self.fc4(x)  # used for classification (either similar or not)
        return x, trans, trans_feat
    def separate(self, x):
        # x shape: [B*2, shape]
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        # concatenate golbal features to local features
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # concatenate local features into global features
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)
        # then use fully connected layer to obtain global features
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = F.relu(self.fc_bn2(self.dropout(self.fc2(x))))
        # below is inspired by
        # https://becominghuman.ai/siamese-networks-algorithm-applications-and-pytorch-implementation-4ffa3304c18
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_reguliarzer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_reguliarzer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
