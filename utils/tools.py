from typing import Any, Literal, Optional

import numpy as np
import open3d as o3d
import vtk
from numpy.typing import NDArray
from vedo import *

def GetVTKTransformationMatrix(
    rotate_X=[-180, 180],
    rotate_Y=[-180, 180],
    rotate_Z=[-180, 180],
    translate_X=[-10, 10],
    translate_Y=[-10, 10],
    translate_Z=[-10, 10],
    scale_X=[0.8, 1.2],
    scale_Y=[0.8, 1.2],
    scale_Z=[0.8, 1.2],
):
    """
    get transformation matrix (4*4)
    return: vtkMatrix4x4
    """
    Trans = vtk.vtkTransform()

    ry_flag = np.random.randint(0, 2)  # if 0, no rotate
    rx_flag = np.random.randint(0, 2)  # if 0, no rotate
    rz_flag = np.random.randint(0, 2)  # if 0, no rotate
    if ry_flag == 1:
        # rotate along Yth axis
        Trans.RotateY(np.random.uniform(rotate_Y[0], rotate_Y[1]))
    if rx_flag == 1:
        # rotate along Xth axis
        Trans.RotateX(np.random.uniform(rotate_X[0], rotate_X[1]))
    if rz_flag == 1:
        # rotate along Zth axis
        Trans.RotateZ(np.random.uniform(rotate_Z[0], rotate_Z[1]))

    trans_flag = np.random.randint(0, 2)  # if 0, no translate
    if trans_flag == 1:
        Trans.Translate(
            [
                np.random.uniform(translate_X[0], translate_X[1]),
                np.random.uniform(translate_Y[0], translate_Y[1]),
                np.random.uniform(translate_Z[0], translate_Z[1]),
            ]
        )

    scale_flag = np.random.randint(0, 2)
    if scale_flag == 1:
        Trans.Scale(
            [
                np.random.uniform(scale_X[0], scale_X[1]),
                np.random.uniform(scale_Y[0], scale_Y[1]),
                np.random.uniform(scale_Z[0], scale_Z[1]),
            ]
        )

    matrix = Trans.GetMatrix()

    return matrix

def normalize(
    input_pts: NDArray[Any],
    type: str = Literal["xyz", "cell_centers"],
    ref: Optional[NDArray[Any]] = None,
):
    # [0:3] = "xyz" or "norm_x,norm_y,norm_z";
    if type == "xyz":
        _mean = input_pts.mean(axis=0) if ref is None else ref.mean(axis=0)
        _std = input_pts.std(axis=0) if ref is None else ref.std(axis=0)
        return np.apply_along_axis(lambda x: (x - _mean) / _std, axis=1, arr=input_pts)
    elif type == "cell_centers":
        _min = input_pts.min(axis=0) if ref is None else ref.min(axis=0)
        _max = input_pts.max(axis=0) if ref is None else ref.max(axis=0)
        return np.apply_along_axis(lambda x: (x - _min) / (_max - _min), axis=1, arr=input_pts)
    else:
        raise NotImplementedError

def cleanup(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    tmp = mesh.remove_degenerate_triangles()
    tmp = tmp.remove_duplicated_triangles()
    tmp = tmp.remove_duplicated_vertices()
    tmp = tmp.remove_non_manifold_edges()
    result = tmp.remove_unreferenced_vertices()
    return result

def decimate_o3d(mesh: o3d.geometry, face_count: int = 10000) -> o3d.geometry.TriangleMesh:
    while np.asarray(mesh.triangles).shape[0] != face_count:
        mesh = mesh.subdivide_midpoint(1).simplify_quadric_decimation(face_count)
    return mesh

def count_channels(input_list):
    result = 0
    if "xyz" in input_list:
        result += 3
    if "xyz3" in input_list:
        result += 9
    if "norm" in input_list:
        result += 3
    if "norm3" in input_list:
        result += 9
    return result
