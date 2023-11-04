# Source: https://github.com/limhoyeon/ToothGroupNetwork
# modified
import torch
import torch.nn as nn

from ..build_models import MODELS
from ..sub_models import get_graph_feature

@MODELS.register_module()
class DGCNN(nn.Module):
    def __init__(
        self,
        num_classes=17,
        num_channels=6,
        k=20,
        embed_dims=1024,
        scale=1,
        dropout=0.5,
    ):
        super(DGCNN, self).__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64 * scale)
        self.bn2 = nn.BatchNorm2d(64 * scale)
        self.bn3 = nn.BatchNorm2d(64 * scale)
        self.bn4 = nn.BatchNorm2d(64 * scale)
        self.bn5 = nn.BatchNorm2d(64 * scale)
        self.bn6 = nn.BatchNorm1d(embed_dims * scale)
        self.bn7 = nn.BatchNorm1d(512 * scale)
        self.bn8 = nn.BatchNorm1d(256 * scale)

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels * 2, 64 * scale, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * scale, 64 * scale, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2 * scale, 64 * scale, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64 * scale, 64 * scale, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64 * 2 * scale, 64 * scale, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(192 * scale, embed_dims * scale, kernel_size=1, bias=False),
            self.bn6,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(1216 * scale, 512 * scale, kernel_size=1, bias=False),
            self.bn7,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(512 * scale, 256 * scale, kernel_size=1, bias=False),
            self.bn8,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.dp1 = nn.Dropout(dropout)
        self.cls_conv = nn.Conv1d(256, num_classes, kernel_size=1, bias=False)
    def forward(self, x):
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, embed_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, embed_dims, num_points) -> (batch_size, embed_dims, 1)

        x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)

        x = self.cls_conv(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        x = x.transpose(2, 1)
        return x
