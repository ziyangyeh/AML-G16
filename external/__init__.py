import os
import sys

from .dgcnn_pytorch.model import DGCNN_semseg_scannet, get_graph_feature
from .MeshSegNet.meshsegnet import MeshSegNet
from .Pointnet_Pointnet2_pytorch.models.pointnet2_sem_seg import get_model as PointNet2
from .Pointnet_Pointnet2_pytorch.models.pointnet2_sem_seg_msg import (
    get_model as PointNet2_msg,
)
from .Pointnet_Pointnet2_pytorch.models.pointnet2_utils import (
    PointNetSetAbstraction,
    PointNetSetAbstractionMsg,
)
from .Pointnet_Pointnet2_pytorch.models.pointnet_sem_seg import get_model as PointNet
from .Pointnet_Pointnet2_pytorch.models.pointnet_utils import PointNetEncoder

sys.path.append(os.getcwd() + "/external/iMeshSegNet/models")
from imeshsegnet import iMeshSegNet
