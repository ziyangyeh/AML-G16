import os
import sys
from argparse import ArgumentParser
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import vedo
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, sys.path[0] + "/../")
from utils import generate_cell_labels_from_point_label, get_fdi_label_map

def merge(obj, info, output_dir: str, fdi_map: dict):
    _jaw, _id = str(info).split("/")[1:-1]
    mesh = vedo.Mesh(str(obj))
    info_df = pd.read_json(info)
    fdi_v = info_df["labels"]
    label_v = info_df["instances"]
    mesh.pointdata["FDI"] = fdi_v
    mesh.pointdata["InsLabels"] = label_v
    mesh.pointdata["Labels"] = np.vectorize(fdi_map.__getitem__)(fdi_v)
    mesh.celldata["FDI"] = (fdi_f := generate_cell_labels_from_point_label(mesh, FDI=True))
    mesh.celldata["InsLabels"] = generate_cell_labels_from_point_label(mesh, FDI=False)
    mesh.celldata["Labels"] = np.vectorize(fdi_map.__getitem__)(fdi_f)
    if os.path.exists(out_dir := f"{output_dir}/{_id}/{_jaw}/"):
        mesh.write(out_dir + "mesh.vtp")
    else:
        os.makedirs(out_dir)
        mesh.write(out_dir + "mesh.vtp")

if __name__ == "__main__":
    parser = ArgumentParser(description="Merge into vtp file.")
    parser.add_argument("-out", "--output", type=str, help="Output Directory", default="formatted_data")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    DATASET_DIR = Path("Teeth3DS")
    LOWER_DIR = DATASET_DIR / Path("lower")
    UPPER_DIR = DATASET_DIR / Path("upper")
    total_pairs = zip(DATASET_DIR.glob("*/*/*.obj"), DATASET_DIR.glob("*/*/*.json"))
    assert (total_len := len(list(DATASET_DIR.glob("*/*/*.obj")))) == len(list(DATASET_DIR.glob("*/*/*.json")))

    fdi_map = get_fdi_label_map()

    Parallel(n_jobs=cpu_count())(delayed(merge)(obj, info, args.output, fdi_map) for obj, info in tqdm(total_pairs, total=total_len))
