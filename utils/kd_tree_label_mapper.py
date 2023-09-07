import gc

import numpy as np
import open3d as o3d
import vedo

def get_index_from_o3d_kdtree(xyz, kd_tree) -> int:
    """Use for downsampling labels"""
    _, idx, _ = kd_tree.search_knn_vector_3d(xyz, 1)
    return idx[0]

def kd_label_mapper(src: vedo.Mesh, dst: o3d.geometry.TriangleMesh) -> vedo.Mesh:
    # Downsample the pointcloud and simplify the mesh to speedup visualization.
    src_pcd_tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np.asarray(src.points()))))
    src_pcd_cc_tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np.asarray(src.cell_centers()))))

    dst_pcd = o3d.geometry.PointCloud(points=dst.vertices)
    dst_cc_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np.asarray(vedo.utils.open3d2vedo(dst).cell_centers())))

    pts_idx = np.apply_along_axis(get_index_from_o3d_kdtree, 1, np.asarray(dst_pcd.points), src_pcd_tree)
    face_idx = np.apply_along_axis(get_index_from_o3d_kdtree, 1, np.asarray(dst_cc_pcd.points), src_pcd_cc_tree)

    del src_pcd_tree
    del src_pcd_cc_tree
    del dst_pcd
    del dst_cc_pcd
    result = vedo.utils.open3d2vedo(dst)
    result.celldata["FDI"] = src.celldata["FDI"][face_idx]
    result.celldata["Labels"] = src.celldata["Labels"][face_idx]
    result.pointdata["FDI"] = src.pointdata["FDI"][pts_idx]
    result.pointdata["Labels"] = src.pointdata["Labels"][pts_idx]
    del pts_idx
    del face_idx
    gc.collect()
    return result
