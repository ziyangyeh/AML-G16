import os
import sys
from argparse import ArgumentParser
from multiprocessing import cpu_count
from pathlib import Path

import cv2
import numpy as np
import vedo
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, sys.path[0] + "/../")
from utils import get_6_faces_rotation_matrix, render_image_and_depth

def generate_image_and_depth(mesh_file, rot_mat_lst: list, **kwargs):
    mesh = vedo.utils.vedo2open3d(vedo.Mesh(str(mesh_file)))
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    for idx, (seq, angle) in enumerate(rot_mat_lst):
        image = render_image_and_depth(mesh, seq, angle, kwargs)
        np.save(os.path.dirname(mesh_file) + f"/depth_{idx}.npy", image["depth"])
        cv2.imwrite(os.path.dirname(mesh_file) + f"/image_{idx}.png", image["image"])

if __name__ == "__main__":
    parser = ArgumentParser(description="Caputure images and depthes from mesh.")
    parser.add_argument("-in", "--input", type=str, help="Input Directory", default="formatted_data")
    parser.add_argument("-w", "--width", type=int, help="Windows Width", default=512)
    parser.add_argument("-he", "--height", type=int, help="Windows height", default=None)

    args = parser.parse_args()

    DATASET_DIR = Path(args.input)
    vtp_lst = list(DATASET_DIR.glob("*/*/mesh_d.vtp"))

    rmat_lst = get_6_faces_rotation_matrix()

    Parallel(n_jobs=cpu_count())(
        delayed(generate_image_and_depth)(vtp_file, rmat_lst, width=args.width, height=args.height) for vtp_file in tqdm(vtp_lst, total=len(vtp_lst))
    )
