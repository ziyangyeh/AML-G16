import sys
from typing import Callable, List, Literal, Optional

import numpy as np
import pandas as pd
import torch
import vedo
from PIL import Image
from einops import rearrange
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
import albumentations as A

sys.path.insert(0, sys.path[0] + "/../")
from utils import normalize

class Teeth3DS(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        jaw: Literal["upper", "lower", "both"] = "both",
        image: bool = True,
        depth: bool = True,
        decimated: bool = True,
        mesh_feature_select: List[Literal["xyz", "xyz3", "norm", "norm3"]] = ["xyz"],
        number_classes: int = 17,  # Gum(1) + teeth(7*2) + wisdom teeh(1*2)
        mesh_aug: Optional[Callable] = None,
        image_aug: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super(Teeth3DS, self).__init__()
        self.dataframe = self._filter(dataframe, jaw, image, depth, decimated)
        self.image = image
        self.depth = depth
        self.rgbd = image and depth
        self.transform = mesh_aug
        self.image_aug = image_aug if image_aug else A.Normalize()
        self.c, self.cols = self._select()
        self.number_classes = number_classes
        self.xyz = True if "xyz" in mesh_feature_select else False
        self.xyz3 = True if "xyz3" in mesh_feature_select else False
        self.norm = True if "norm" in mesh_feature_select else False
        self.norm3 = True if "norm3" in mesh_feature_select else False
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, index):
        # IMAGE PIPELINE
        image_dict = {}
        if self.c == 4:
            for idx, (_image, _depth) in enumerate(self.cols):
                cur_img, cur_dep = self.dataframe.iloc[
                    index,
                    [
                        self.dataframe.columns.get_loc(_image),
                        self.dataframe.columns.get_loc(_depth),
                    ],
                ]
                img = Image.open(cur_img)
                depth = np.load(cur_dep)
                auged = self.image_aug(image=np.asarray(img), mask=depth)
                img = np.concatenate(
                    [
                        auged["image"],
                        np.expand_dims(auged["mask"], axis=-1),
                    ],
                    axis=-1,
                )
                image_dict[f"image_{idx}"] = img
        elif self.c == 3 or self.c == 1:
            for idx, _item in enumerate(self.cols):
                cur = self.dataframe.iloc[index, self.dataframe.columns.get_loc(_item)]
                img = np.asarray(Image.open(cur)) if self.c == 3 else np.load(cur)
                if self.c != 1:
                    auged = self.image_aug(image=img)
                    image_dict[f"image_{idx}"] = auged["image"]
                else:
                    image_dict[f"image_{idx}"] = img

        # MESH PIPELINE
        mesh = vedo.Mesh(self.dataframe.iloc[index, self.dataframe.columns.get_loc("mesh")])

        if self.transform:
            mesh = self.transform(mesh)

        # Traslate into origin.
        mesh.points(mesh.points() - mesh.center_of_mass())
        # Get translated points
        points = mesh.points()

        feature_list = []
        if self.xyz:
            # xyz in "feat"
            cell_centers = mesh.cell_centers()
            cell_centers = normalize(cell_centers, ref=points, type="cell_centers")
            feature_list.append(cell_centers)

        if self.xyz3:
            # x0,y0,z0,x1,y1,z1,x2,y2,z2 in "feat", Ncells * 3 * 3 ==> Necells * 9
            faces_vertices = points[mesh.faces()].reshape(mesh.ncells * 3, 3)
            # Normalize cells' 3 vertices(from cells' indices).
            faces_vertices = normalize(faces_vertices, ref=points, type="xyz").reshape(mesh.ncells, 9)
            feature_list.append(faces_vertices)

        if self.norm:
            # Compute normals
            mesh.compute_normals()
            cell_normals = mesh.celldata["Normals"]
            cell_normals = normalize(cell_normals, type="xyz")
            feature_list.append(cell_normals)

        if self.norm3:
            faces_vertices_normals = mesh.pointdata["Normals"][mesh.faces()].reshape(mesh.ncells * 3, 3)
            faces_vertices_normals = normalize(faces_vertices_normals, type="xyz").reshape(mesh.ncells, 9)
            feature_list.append(faces_vertices_normals)

        # Concatenate features
        feats = np.column_stack(feature_list)

        # Labels to one-hot
        labels = torch.tensor(mesh.celldata["Labels"], device=torch.device("cpu"))
        onehot = one_hot(labels, num_classes=self.number_classes).numpy()

        # Combine modalities
        if self.c != 0:
            if self.c != 1:
                images = rearrange(np.asarray(list(image_dict.values())), "n h w c -> n c h w")
            else:
                images = rearrange(np.asarray(list(image_dict.values())), "n h w-> n () h w")
            return {"x": feats.astype(np.float32), "labels": labels, "onehot": onehot, "images": images}
        return {
            "x": feats.astype(np.float32),
            "labels": labels,
            "onehot": onehot,
        }
    def _select(self):
        if self.image:
            img_lst = sorted([i for i in self.dataframe.columns.values if "image" in i])
        if self.depth:
            depth_lst = sorted([i for i in self.dataframe.columns.values if "depth" in i])
        if self.rgbd:
            return 4, list(zip(img_lst, depth_lst))
        elif self.image or self.depth:
            if not self.image:
                return 1, depth_lst
            elif not self.depth:
                return 3, img_lst
        else:
            return 0, None
    def _filter(
        self,
        dataframe: pd.DataFrame,
        jaw: str,
        image: bool,
        depth: bool,
        decimated: bool,
    ):
        if jaw != "both":
            selected = list(filter(lambda x: jaw in x, dataframe.columns.values))
            if not image:
                selected = [i for i in selected if "image" not in i]
            if not depth:
                selected = [i for i in selected if "depth" not in i]
            result = dataframe[selected].rename(columns={k: v for k, v in zip(selected, [i[6:] for i in selected])})
        else:
            upper = list(filter(lambda x: "upper" in x, dataframe.columns.values))
            lower = list(filter(lambda x: "lower" in x, dataframe.columns.values))
            if not image:
                upper = [i for i in upper if "image" not in i]
                lower = [i for i in lower if "image" not in i]
            if not depth:
                upper = [i for i in upper if "depth" not in i]
                lower = [i for i in lower if "depth" not in i]
            upper_df = dataframe[upper].reset_index(drop=True).rename(columns=({k: v for k, v in zip(upper, [i[6:] for i in upper])}))
            lower_df = dataframe[lower].reset_index(drop=True).rename(columns=({k: v for k, v in zip(lower, [i[6:] for i in lower])}))
            result = pd.concat([upper_df, lower_df], ignore_index=True, axis=0)
        if decimated:
            result.drop(columns=["mesh"], inplace=True)
            result.rename(columns={"mesh_d": "mesh"}, inplace=True)
        return result

if __name__ == "__main__":
    df = pd.read_csv("formatted_data/teeth3ds.csv")
    teeth3ds = Teeth3DS(df)
    print({f"{k}_shape": v.shape for k, v in teeth3ds[0].items()})
    teeth3ds = Teeth3DS(df, image=False)
    print({f"{k}_shape": v.shape for k, v in teeth3ds[0].items()})
    teeth3ds = Teeth3DS(df, "lower", depth=False)
    print({f"{k}_shape": v.shape for k, v in teeth3ds[0].items()})
    teeth3ds = Teeth3DS(df, "upper", mesh_feature_select=["xyz", "xyz3", "norm", "norm3"])
    print({f"{k}_shape": v.shape for k, v in teeth3ds[0].items()})
    teeth3ds = Teeth3DS(df, "upper", mesh_feature_select=["xyz", "xyz3", "norm"])
    print({f"{k}_shape": v.shape for k, v in teeth3ds[0].items()})
    teeth3ds = Teeth3DS(df, "upper", mesh_feature_select=["norm3", "norm"])
    print({f"{k}_shape": v.shape for k, v in teeth3ds[0].items()})
    teeth3ds = Teeth3DS(df, "lower", image=False, depth=False)
    print({f"{k}_shape": v.shape for k, v in teeth3ds[0].items()})
