from typing import Any, Literal, Optional

import numpy as np
import open3d as o3d
from numpy.typing import NDArray


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
        return np.apply_along_axis(
            lambda x: (x - _min) / (_max - _min), axis=1, arr=input_pts
        )
    else:
        raise NotImplementedError


def cleanup(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    tmp = mesh.remove_degenerate_triangles()
    tmp = tmp.remove_duplicated_triangles()
    tmp = tmp.remove_duplicated_vertices()
    tmp = tmp.remove_non_manifold_edges()
    result = tmp.remove_unreferenced_vertices()
    return result


def decimate_o3d(
    mesh: o3d.geometry, face_count: int = 10000
) -> o3d.geometry.TriangleMesh:
    while np.asarray(mesh.triangles).shape[0] != face_count:
        mesh = mesh.subdivide_midpoint(1).simplify_quadric_decimation(face_count)
    return mesh
