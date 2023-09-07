from typing import Any

import cv2
import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

def get_6_faces_rotation_matrix():
    rot_params = {"x": [0, 90, 180, 270], "y": [90, 180]}
    lst = []
    for k in rot_params.keys():
        for v in rot_params[k]:
            lst.append((k, v))
    return lst

def render_image_and_depth(
    input_mesh: o3d.geometry.TriangleMesh,
    seq: str = "x",  # x,y,z,xyz,zyx
    angle: int = 0,  # 0-360
    degree: bool = True,
    width: int = 512,
    height: int = None,
    material: str = "defaultLitTransparency",
) -> dict[str, NDArray[Any]]:
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height if height else width)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = material

    rot_mat = R.from_euler(seq, angle, degree).as_matrix().astype(np.int8)
    renderer.scene.set_background(np.array([0, 0, 0, 0]))
    renderer.scene.add_geometry("mesh", input_mesh.rotate(rot_mat), mat)

    depth = renderer.render_to_depth_image()
    depth = np.asarray(depth, dtype=np.float64)
    # Create a copy of the loaded data
    depth2 = depth.copy()
    # Replace 0 values in the copied data with the second unique value
    depth2[depth2 == 0] = np.unique(depth)[1]
    # Normalize the copied data to the range [0, 1]
    depth2 = cv2.normalize(depth2, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Replace 0 values with -1 and other values with corresponding normalized_data values
    depth2[depth == 0] = -1

    image = renderer.render_to_image()
    del renderer
    return {"image": np.asarray(image), "depth": depth2}
