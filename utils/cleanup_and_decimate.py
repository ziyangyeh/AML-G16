import numpy as np
import open3d as o3d


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
