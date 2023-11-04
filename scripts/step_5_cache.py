import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vedo
from scipy.spatial import distance_matrix
from tqdm import tqdm

sys.path.insert(0, sys.path[0] + "/../")
from utils import get_graph_feature_cpu, normalize

CSV_PATH = Path("formatted_data/teeth3ds.csv")

df = pd.read_csv(CSV_PATH)

u_kg6_lst = []
u_kg12_lst = []
u_as_lst = []
u_al_lst = []
l_kg6_lst = []
l_kg12_lst = []
l_as_lst = []
l_al_lst = []
for i in tqdm(range(len(df))):
    lower, upper = df.loc[i, ["lower_mesh_d", "upper_mesh_d"]].values
    for pos, j in [("lower", lower), ("upper", upper)]:
        mesh = vedo.Mesh(j)
        # Traslate into origin.
        mesh.points(mesh.points() - mesh.center_of_mass())
        points = mesh.points()
        cell_centers = mesh.cell_centers()
        cell_centers = normalize(cell_centers, ref=points, type="cell_centers")

        S1 = np.zeros([len(cell_centers), len(cell_centers)], dtype="float32")
        S2 = np.zeros([len(cell_centers), len(cell_centers)], dtype="float32")
        D = distance_matrix(cell_centers, cell_centers)
        S1[D < 0.1] = 1.0
        S1 = S1 / np.dot(np.sum(S1, axis=1, keepdims=True), np.ones((1, len(cell_centers))))

        S2[D < 0.2] = 1.0
        S2 = S2 / np.dot(np.sum(S2, axis=1, keepdims=True), np.ones((1, len(cell_centers))))

        KG_6 = get_graph_feature_cpu(cell_centers.transpose(1, 0), k=6)
        KG_12 = get_graph_feature_cpu(cell_centers.transpose(1, 0), k=12)

        kg_6_p = os.path.dirname(j) + "/kg6.npy"
        kg_12_p = os.path.dirname(j) + "/kg12.npy"
        a_s_p = os.path.dirname(j) + "/as.npy"
        a_l_p = os.path.dirname(j) + "/al.npy"
        np.save(kg_6_p, KG_6)
        np.save(kg_12_p, KG_12)
        np.save(a_s_p, S1)
        np.save(a_l_p, S2)
        if pos == "lower":
            l_kg6_lst.append(kg_6_p)
            l_kg12_lst.append(kg_12_p)
            l_as_lst.append(a_s_p)
            l_al_lst.append(a_l_p)
        elif pos == "upper":
            u_kg6_lst.append(kg_6_p)
            u_kg12_lst.append(kg_12_p)
            u_as_lst.append(a_s_p)
            u_al_lst.append(a_l_p)

df["lower_kg6"] = l_kg6_lst
df["lower_kg12"] = l_kg12_lst
df["lower_as"] = l_as_lst
df["lower_al"] = l_al_lst
df["upper_kg6"] = u_kg6_lst
df["upper_kg12"] = u_kg12_lst
df["upper_as"] = u_as_lst
df["upper_al"] = u_al_lst
df.to_csv("formatted_data/teeth3ds.csv", index=False)
