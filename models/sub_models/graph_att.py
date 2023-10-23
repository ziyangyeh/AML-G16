import torch
import torch.nn as nn
import torch.nn.functional as F

from .helper import index_points

class GraphAttention(nn.Module):
    def __init__(self, feature_dim, out_dim, K):
        super(GraphAttention, self).__init__()
        self.dropout = 0.6
        self.conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.K = K
    def forward(self, Graph_index, x, feature):
        B, C, N = x.shape
        x = x.contiguous().view(B, N, C)
        feature = feature.permute(0, 2, 3, 1)
        neighbor_feature = index_points(x, Graph_index)
        centre = x.view(B, N, 1, C).expand(B, N, self.K, C)
        delta_f = torch.cat([centre - neighbor_feature, neighbor_feature], dim=3).permute(0, 3, 2, 1)
        e = self.conv(delta_f)
        e = e.permute(0, 3, 2, 1)
        attention = F.softmax(e, dim=2)  # [B, npoint, nsample,D]
        graph_feature = torch.sum(torch.mul(attention, feature), dim=2).permute(0, 2, 1)
        return graph_feature
