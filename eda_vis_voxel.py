import copy
import ctypes
import gc
import time

# %%
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

DATASET_DIR = Path("Teeth3DS")

LOWER_DIR = DATASET_DIR / Path("lower")
UPPER_DIR = DATASET_DIR / Path("upper")

total_pairs = zip(DATASET_DIR.glob("*/*/*.obj"), DATASET_DIR.glob("*/*/*.json"))
assert len(tmp_lst := list(DATASET_DIR.glob("*/*/*.obj"))) == len(
    list(DATASET_DIR.glob("*/*/*.json"))
)
total_len = len(tmp_lst)

lower_pairs = zip(LOWER_DIR.glob("*/*.obj"), LOWER_DIR.glob("*/*.json"))
assert len(tmp_lst := list(LOWER_DIR.glob("*/*.obj"))) == len(
    list(LOWER_DIR.glob("*/*.json"))
)
lower_len = len(tmp_lst)

upper_pairs = zip(UPPER_DIR.glob("*/*.obj"), UPPER_DIR.glob("*/*.json"))
assert len(tmp_lst := list(UPPER_DIR.glob("*/*.obj"))) == len(
    list(UPPER_DIR.glob("*/*.json"))
)
upper_len = len(tmp_lst)

import numpy as np
import vedo

# %%
import vtk


def get_point_ids_by_point_label(mesh, label_num=0, FDI=False):
    labels = mesh.pointdata["Labels"] if not FDI else mesh.pointdata["FDI"]
    index = np.where(labels == label_num)
    return index[0]


def get_cell_ids_by_point_index(mesh, indices):
    cell_ids = vtk.vtkIdList()
    data = mesh.inputdata()
    data.BuildLinks()
    result = []
    for i in np.unique(indices):
        data.GetPointCells(i, cell_ids)
        for j in range(cell_ids.GetNumberOfIds()):
            result.append(cell_ids.GetId(j))
    return result


def get_cell_ids_by_point_label(mesh, label_num=0, FDI=False):
    cell_ids = vtk.vtkIdList()
    data = mesh.inputdata()
    data.BuildLinks()
    indices = get_point_ids_by_point_label(mesh, label_num, FDI)
    result = []
    for i in np.unique(indices):
        data.GetPointCells(i, cell_ids)
        for j in range(cell_ids.GetNumberOfIds()):
            result.append(cell_ids.GetId(j))
    return result


def generate_cell_labels_from_point_label(mesh, FDI=False):
    cell_labels = np.zeros(mesh.ncells, dtype=np.uint8)
    if not FDI:
        for i in range(mesh.pointdata["Labels"].max() + 1):
            cell_labels[get_cell_ids_by_point_label(mesh, i, FDI)] = i
    else:
        for i in np.unique(mesh.pointdata["FDI"]):
            cell_labels[get_cell_ids_by_point_label(mesh, i, FDI)] = i
    return cell_labels


def crop_cells_by_point_label(mesh: vedo.Mesh, label_num=0, FDI=False):
    labels = mesh.pointdata["Label"] if not FDI else mesh.pointdata["FDI"]
    ids = np.where(labels != label_num)
    return mesh.clone(deep=True).delete_cells_by_point_index(ids)


# %%
obj, info = list(total_pairs)[0]

import pandas as pd

# %%
import trimesh

mesh = vedo.Mesh(str(obj))
info_df = pd.read_json(str(info))
fdi_label_map = dict(
    zip(
        (
            tmp_group := info_df.copy()
            .groupby(["labels", "instances"])
            .size()
            .reset_index()
        )["labels"],
        tmp_group["instances"],
    )
)
label_fdi_map = dict(zip(tmp_group["instances"], tmp_group["labels"]))
mesh.pointdata["FDI"] = info_df["labels"]
mesh.pointdata["Labels"] = info_df["instances"]
mesh.celldata["Labels"] = generate_cell_labels_from_point_label(mesh)
mesh.celldata["FDI"] = generate_cell_labels_from_point_label(mesh, FDI=True)

# %%
fdi = np.unique(mesh.pointdata["FDI"])
assert fdi[0] == 0
quad1 = np.array([i for i in fdi if i // 10 == 1])
quad2 = np.array([i for i in fdi if i // 10 == 2])
quad3 = np.array([i for i in fdi if i // 10 == 3])
quad4 = np.array([i for i in fdi if i // 10 == 4])
if len(quad3) == 0 and len(quad4) == 0:
    arranged = np.concatenate([np.array([0]), np.flip(quad2), quad1])
elif len(quad1) == 0 and len(quad2) == 0:
    arranged = np.concatenate([np.array([0]), np.flip(quad4), quad3])
else:
    raise Exception("Check metadata!")


# %%
def remap(arr, cdict: dict) -> int:
    """Trimesh use only (for visualization)"""
    for k, v in cdict.items():
        if (v == arr).all():
            return k
    return 0


import plotly.colors as pc

# %%
import plotly.graph_objects as go

cmap = pc.qualitative.Light24
gum_color = "#A0A0A0"

# assert (
#     len(cmap) > len(arranged) - 1
# ), "Cannot assign each label type with a unique color."

fdi_cmap_dict = {
    fdi: trimesh.visual.color.hex_to_rgba(cmap[idx % len(cmap)])
    for idx, fdi in enumerate(arranged[1:])
}
fdi_cmap_dict[0] = trimesh.visual.color.hex_to_rgba(gum_color)
fdi_pts_color = np.array([fdi_cmap_dict[i] for i in mesh.pointdata["FDI"]])
fdi_pts_color = fdi_pts_color.astype(np.float64)[:, :-1]
fdi_pts_color /= 255.0
mesh_o = vedo.utils.vedo2open3d(mesh)
mesh_o.compute_vertex_normals()
mesh_o.compute_triangle_normals()
mesh_o.vertex_colors = o3d.utility.Vector3dVector(fdi_pts_color)

pcd = o3d.geometry.PointCloud(points=mesh_o.vertices)
pcd.estimate_normals()
pcd.colors = mesh_o.vertex_colors

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.75)
o3d.visualization.draw_geometries([voxel_grid])
