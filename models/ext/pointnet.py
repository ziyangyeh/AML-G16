# Source: https://github.com/limhoyeon/ToothGroupNetwork
# modified
import torch
import torch.nn as nn
import torch.nn.functional as F

from openpoints.models.backbone.pointnet import STN3d, STNkd

from ..build_models import MODELS

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3, scale=1):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.scale = scale
        self.conv1 = torch.nn.Conv1d(channel, 64 * scale, 1)
        self.conv2 = torch.nn.Conv1d(64 * scale, 128 * scale, 1)
        self.conv3 = torch.nn.Conv1d(128 * scale, 1024 * scale, 1)
        self.bn1 = nn.BatchNorm1d(64 * scale)
        self.bn2 = nn.BatchNorm1d(128 * scale)
        self.bn3 = nn.BatchNorm1d(1024 * scale)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64 * scale)
    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024 * self.scale)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024 * self.scale, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

@MODELS.register_module()
class PointNet(nn.Module):
    def __init__(self, num_classes=17, num_channels=6, scale=2, dropout=0.5):
        super(PointNet, self).__init__()
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=num_channels, scale=scale)
        self.conv1 = nn.Conv1d(1088 * scale, 512 * scale, 1)
        self.conv2 = nn.Conv1d(512 * scale, 256 * scale, 1)
        self.conv3 = nn.Conv1d(256 * scale, 128 * scale, 1)
        self.conv4 = nn.Conv1d(128 * scale, num_classes, 1)
        self.bn1 = nn.BatchNorm1d(512 * scale)
        self.bn2 = nn.BatchNorm1d(256 * scale)
        self.bn3 = nn.BatchNorm1d(128 * scale)
        self.dropout = nn.Dropout(dropout if dropout else 0.0)
    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1)
        return x
