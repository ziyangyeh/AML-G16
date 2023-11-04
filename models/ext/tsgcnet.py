import torch
import torch.nn as nn

from openpoints.models.backbone.pointnet import STNkd

from ..build_models import MODELS
from ..sub_models.graph_att import GraphAttention
from ..sub_models.pair_knn import get_graph_feature_pair as get_graph_feature

@MODELS.register_module()
class TSGCNet(nn.Module):
    def __init__(self, k=16, num_channels=12, num_classes=17, dropout=0.5):
        super(TSGCNet, self).__init__()
        self.k = k
        """ coordinate stream """
        self.bn1_c = nn.BatchNorm2d(64)
        self.bn2_c = nn.BatchNorm2d(128)
        self.bn3_c = nn.BatchNorm2d(256)
        self.bn4_c = nn.BatchNorm1d(512)
        self.conv1_c = nn.Sequential(
            nn.Conv2d(num_channels * 2, 64, kernel_size=1, bias=False),
            self.bn1_c,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv2_c = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn2_c,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv3_c = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn3_c,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv4_c = nn.Sequential(
            nn.Conv1d(448, 512, kernel_size=1, bias=False),
            self.bn4_c,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.attention_layer1_c = GraphAttention(feature_dim=12, out_dim=64, K=self.k)
        self.attention_layer2_c = GraphAttention(feature_dim=64, out_dim=128, K=self.k)
        self.attention_layer3_c = GraphAttention(feature_dim=128, out_dim=256, K=self.k)
        self.FTM_c1 = STNkd(k=12)
        """ normal stream """
        self.bn1_n = nn.BatchNorm2d(64)
        self.bn2_n = nn.BatchNorm2d(128)
        self.bn3_n = nn.BatchNorm2d(256)
        self.bn4_n = nn.BatchNorm1d(512)
        self.conv1_n = nn.Sequential(
            nn.Conv2d((num_channels) * 2, 64, kernel_size=1, bias=False),
            self.bn1_n,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv2_n = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn2_n,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv3_n = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn3_n,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv4_n = nn.Sequential(
            nn.Conv1d(448, 512, kernel_size=1, bias=False),
            self.bn4_n,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.FTM_n1 = STNkd(k=12)

        """feature-wise attention"""

        self.fa = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )

        """ feature fusion """
        self.pred1 = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.pred2 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.pred3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.pred4 = nn.Sequential(nn.Conv1d(128, num_classes, kernel_size=1, bias=False))
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.dp3 = nn.Dropout(dropout)
    def forward(self, x):
        coor = x[:, :12, :]
        nor = x[:, 12:, :]

        # transform
        trans_c = self.FTM_c1(coor)
        coor = coor.transpose(2, 1)
        coor = torch.bmm(coor, trans_c)
        coor = coor.transpose(2, 1)
        trans_n = self.FTM_n1(nor)
        nor = nor.transpose(2, 1)
        nor = torch.bmm(nor, trans_n)
        nor = nor.transpose(2, 1)

        coor1, nor1, index = get_graph_feature(coor, nor, k=self.k)
        coor1 = self.conv1_c(coor1)
        nor1 = self.conv1_n(nor1)
        coor1 = self.attention_layer1_c(index, coor, coor1)
        nor1 = nor1.max(dim=-1, keepdim=False)[0]

        coor2, nor2, index = get_graph_feature(coor1, nor1, k=self.k)
        coor2 = self.conv2_c(coor2)
        nor2 = self.conv2_n(nor2)
        coor2 = self.attention_layer2_c(index, coor1, coor2)
        nor2 = nor2.max(dim=-1, keepdim=False)[0]

        coor3, nor3, index = get_graph_feature(coor2, nor2, k=self.k)
        coor3 = self.conv3_c(coor3)
        nor3 = self.conv3_n(nor3)
        coor3 = self.attention_layer3_c(index, coor2, coor3)
        nor3 = nor3.max(dim=-1, keepdim=False)[0]

        coor = torch.cat((coor1, coor2, coor3), dim=1)
        coor = self.conv4_c(coor)
        nor = torch.cat((nor1, nor2, nor3), dim=1)
        nor = self.conv4_n(nor)
        avgSum_coor = coor.mean(1)
        avgSum_nor = nor.mean(1)
        avgSum = avgSum_coor + avgSum_nor
        weight_coor = (avgSum_coor / avgSum).unsqueeze(1)
        weight_nor = (avgSum_nor / avgSum).unsqueeze(1)
        x = torch.cat((coor * weight_coor, nor * weight_nor), dim=1)

        weight = self.fa(x)
        x = weight * x

        x = self.pred1(x)
        self.dp1(x)
        x = self.pred2(x)
        self.dp2(x)
        x = self.pred3(x)
        self.dp3(x)
        score = self.pred4(x)
        score = score.permute(0, 2, 1)
        return score
