import os
import sys

sys.path.append(os.getcwd() + "/external/iMeshSegNet/models")
from imeshsegnet import iMeshSegNet

from .dgcnn_pytorch.model import DGCNN_partseg, DGCNN_semseg_s3dis, DGCNN_semseg_scannet
from .MeshSegNet.meshsegnet import MeshSegNet
from .Pointnet_Pointnet2_pytorch.models.pointnet2_sem_seg import get_model as PointNet2
from .Pointnet_Pointnet2_pytorch.models.pointnet2_sem_seg_msg import (
    get_model as PointNet2_msg,
)
from .Pointnet_Pointnet2_pytorch.models.pointnet_sem_seg import get_model as PointNet
