import os
import sys
from argparse import Namespace
import torch

from .Pointnet_Pointnet2_pytorch.models.pointnet_utils import PointNetEncoder
from .Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstraction
from .dgcnn_pytorch.model import DGCNN_partseg, DGCNN_semseg_s3dis, DGCNN_semseg_scannet
from .MeshSegNet.meshsegnet import MeshSegNet
from .Pointnet_Pointnet2_pytorch.models.pointnet2_sem_seg import get_model as PointNet2
from .Pointnet_Pointnet2_pytorch.models.pointnet2_sem_seg_msg import (
    get_model as PointNet2_msg,
)
from .Pointnet_Pointnet2_pytorch.models.pointnet_sem_seg import get_model as PointNet
from .registry import Registry

sys.path.append(os.getcwd() + "/external/iMeshSegNet/models")
from imeshsegnet import iMeshSegNet

EXT_MODELS = Registry("models", scope="ext")

@EXT_MODELS.register_module()
class PointNet(PointNet):
    def __init__(self, **kwargs):
        args = Namespace(**kwargs)
        super(PointNet, self).__init__(args.num_classes)
        if "k" in kwargs.keys():
            self.k = kwargs["k"]
            self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=kwargs["channel"])

@EXT_MODELS.register_module()
class PointNet2(PointNet2):
    def __init__(self, **kwargs):
        args = Namespace(**kwargs)
        super(PointNet2, self).__init__(args.num_classes)
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, kwargs["channel"] + 3, [32, 32, 64], False)

@EXT_MODELS.register_module()
class PointNet2_msg(PointNet2_msg):
    def __init__(self, **kwargs):
        args = Namespace(**kwargs)
        super(PointNet2_msg, self).__init__(args.num_classes)
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, kwargs["channel"], [32, 32, 64], False)

@EXT_MODELS.register_module()
class DGCNN_partseg(DGCNN_partseg):
    def __init__(self, **kwargs):
        if not kwargs["xyz"]:
            raise Exception("feats should contain 'xyz' only")
        args = Namespace(**kwargs)
        super(DGCNN_partseg, self).__init__(args, args.num_classes)

@EXT_MODELS.register_module()
class DGCNN_semseg_s3dis(DGCNN_semseg_s3dis):
    def __init__(self, **kwargs):
        if not kwargs["xyz"]:
            raise Exception("feats should contain 'xyz' only")
        args = Namespace(**kwargs)
        super(DGCNN_semseg_s3dis, self).__init__(args, args.num_classes)

@EXT_MODELS.register_module()
class DGCNN_semseg_scannet(DGCNN_semseg_scannet):
    def __init__(self, **kwargs):
        if not kwargs["xyz"]:
            raise Exception("feats should contain 'xyz' only")
        args = Namespace(**kwargs)
        super(DGCNN_semseg_scannet, self).__init__(args.num_classes, args.k, args.emb_dims, args.dropout)

@EXT_MODELS.register_module()
class MeshSegNet(MeshSegNet):
    def __init__(self, **kwargs):
        args = Namespace(**kwargs)
        super(MeshSegNet, self).__init__(
            args.num_classes,
            args.num_channels,
            True if args.dropout else False,
            args.dropout,
        )

@EXT_MODELS.register_module()
class iMeshSegNet(iMeshSegNet):
    def __init__(self, **kwargs):
        args = Namespace(**kwargs)
        super(iMeshSegNet, self).__init__(
            args.num_classes,
            args.num_channels,
            True if args.dropout else False,
            args.dropout,
        )
