from typing import Callable, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class Teeth3DS(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        jaw: str = None,
        mode: str = None,
        transform: Optional[Callable] = None,
    ) -> None:
        super(Teeth3DS, self).__init__()
        self.jaw = jaw
        self.mode = mode
        self.dataframe = (
            pd.DataFrame(
                {
                    "obj": pd.Series(dataframe[f"{jaw}_obj"].to_numpy()),
                    "json": pd.Series(dataframe[f"{jaw}_json"].to_numpy()),
                }
            )
            if jaw
            else pd.DataFrame(
                {
                    "obj": pd.Series(
                        np.concatenate(
                            [
                                dataframe["upper_obj"].to_numpy(),
                                dataframe["lower_obj"].to_numpy(),
                            ]
                        )
                    ),
                    "json": pd.Series(
                        np.concatenate(
                            [
                                dataframe["upper_json"].to_numpy(),
                                dataframe["lower_json"].to_numpy(),
                            ]
                        )
                    ),
                }
            )
        )
        self.transform = transform

    def len(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        print(
            self.dataframe.iloc[
                index,
                [
                    self.dataframe.columns.get_loc("obj"),
                    self.dataframe.columns.get_loc("json"),
                ],
            ]
        )
        return None


if __name__ == "__main__":
    df = pd.read_csv("teeth3ds.csv")
    teeth3ds = Teeth3DS(df, "upper")
    print(teeth3ds[0])
    teeth3ds = Teeth3DS(df, "lower")
    print(teeth3ds[0])
    teeth3ds = Teeth3DS(df)
    print(teeth3ds[0])
