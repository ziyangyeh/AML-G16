import os
import sys

import torch
import torch.nn.functional as F

from utils import Registry

from external import *

EXT_MODELS = Registry("models", scope="ext")

@EXT_MODELS.register_module()
class PointNet(PointNet):
    def __init__(self, **kwargs):
        super(PointNet, self).__init__(kwargs["num_classes"])
        if "k" in kwargs.keys():
            self.k = kwargs["k"]
            self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=kwargs["channel"])

@EXT_MODELS.register_module()
class PointNet2(PointNet2):
    def __init__(self, **kwargs):
        super(PointNet2, self).__init__(kwargs["num_classes"])
        self.sa1 = PointNetSetAbstraction(kwargs["npoints"], 0.1, 32, kwargs["channel"], [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256, [256, 256, 512], False)
    def forward(self, x):
        l0_points = x
        l0_xyz = x[:, :3, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points

@EXT_MODELS.register_module()
class PointNet2_MSG(PointNet2_msg):
    def __init__(self, **kwargs):
        super(PointNet2_MSG, self).__init__(kwargs["num_classes"])
        self.sa1 = PointNetSetAbstractionMsg(kwargs["npoints"], [0.05, 0.1], [16, 32], kwargs["channel"], [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32 + 64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128 + 128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256 + 256, [[256, 256, 512], [256, 384, 512]])
    def forward(self, x):
        l0_points = x
        l0_xyz = x[:, :3, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points

@EXT_MODELS.register_module()
class DGCNN(DGCNN_semseg_scannet):
    def __init__(self, **kwargs):
        if not kwargs["xyz"]:
            raise Exception("feats should contain 'xyz' only")
        super(DGCNN, self).__init__(kwargs["num_classes"], kwargs["k"], kwargs["emb_dims"], kwargs["dropout"])
    def forward(self, x):
        npoint = x.size(2)
        x = get_graph_feature(x, k=self.k, dim9=False)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x2, k=self.k)
        x = self.conv5(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv6(x)
        x = x.max(dim=-1, keepdim=True)[0]
        x = x.repeat(1, 1, npoint)
        x = torch.cat((x, x1, x2, x3), dim=1)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)
        # x = x.transpose(2, 1).contiguous()
        return x, None

@EXT_MODELS.register_module()
class MeshSegNet(MeshSegNet):
    def __init__(self, **kwargs):
        super(MeshSegNet, self).__init__(
            kwargs["num_classes"],
            kwargs["num_channels"],
            True if kwargs["dropout"] else False,
            kwargs["dropout"],
        )

@EXT_MODELS.register_module()
class iMeshSegNet(iMeshSegNet):
    def __init__(self, **kwargs):
        super(iMeshSegNet, self).__init__(
            kwargs["num_classes"],
            kwargs["num_channels"],
            True if kwargs["dropout"] else False,
            kwargs["dropout"],
        )
