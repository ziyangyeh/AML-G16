from pathlib import Path

import pandas as pd

# Fixed path "Teeth3DS" in repository root.
DATASET_DIR = Path("Teeth3DS")

LOWER_DIR = DATASET_DIR / Path("lower")
UPPER_DIR = DATASET_DIR / Path("upper")

total_pairs = zip(DATASET_DIR.glob("*/*/*.obj"), DATASET_DIR.glob("*/*/*.json"))
assert len(tmp_lst := list(DATASET_DIR.glob("*/*/*.obj"))) == len(
    list(DATASET_DIR.glob("*/*/*.json"))
)

lower_pairs = zip(
    lower_obj := LOWER_DIR.glob("*/*.obj"), lower_json := LOWER_DIR.glob("*/*.json")
)
assert len(tmp_lst := list(LOWER_DIR.glob("*/*.obj"))) == len(
    list(LOWER_DIR.glob("*/*.json"))
)

upper_pairs = zip(
    upper_obj := UPPER_DIR.glob("*/*.obj"), upper_json := UPPER_DIR.glob("*/*.json")
)
assert len(tmp_lst := list(UPPER_DIR.glob("*/*.obj"))) == len(
    list(UPPER_DIR.glob("*/*.json"))
)

_t = list(total_pairs)
_dict = {
    str(_obj).split("/")[-2]: {
        "lower_obj": None,
        "lower_json": None,
        "upper_obj": None,
        "upper_json": None,
    }
    for _obj, _ in _t
}
for _obj, _json in _t:
    assert (_id := str(_obj).split("/")[-2]) == str(_json).split("/")[-2]
    assert (_jaw := str(_obj).split("/")[-3]) == str(_json).split("/")[-3]
    _dict[_id][f"{_jaw}_obj"] = str(_obj)
    _dict[_id][f"{_jaw}_json"] = str(_json)

dataset_df = pd.DataFrame().from_dict(_dict, orient="index").reset_index(names="id")
dataset_df.to_csv("teeth3ds.csv", index=False)
