import gc
import os
import sys
from argparse import ArgumentParser
from multiprocessing import cpu_count
from pathlib import Path

import vedo
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, sys.path[0] + "/../")
from utils import cleanup, decimate_o3d, kd_label_mapper

def decimate_with_label(mesh_file, face_count: int = 10000):
    mesh_vedo = vedo.Mesh(str(mesh_file))
    mesh_o3d_sim = decimate_o3d(cleanup(vedo.utils.vedo2open3d(mesh_vedo)), face_count)
    result = kd_label_mapper(mesh_vedo, mesh_o3d_sim)
    result.write(os.path.dirname(mesh_file) + "/mesh_d.vtp")
    del mesh_vedo
    del mesh_o3d_sim
    del result
    gc.collect()

if __name__ == "__main__":
    parser = ArgumentParser(description="Decimate mesh to N cells.")
    parser.add_argument("-n", "--number", type=int, help="Number for decimating", default=10000)

    args = parser.parse_args()

    DATASET_DIR = Path("formatted_data")
    vtp_lst = list(DATASET_DIR.glob("*/*/*.vtp"))

    # loky backend causes memory leak.
    Parallel(n_jobs=cpu_count(), backend="multiprocessing")(delayed(decimate_with_label)(vtp_file, args.number) for vtp_file in tqdm(vtp_lst, total=len(vtp_lst)))
