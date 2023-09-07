from pathlib import Path

import pandas as pd

DATASET_DIR = Path("formatted_data")

ID_GLOB = list(DATASET_DIR.glob("*/*/*"))

result = {}
for item in ID_GLOB:
    _id = (splited := str(item).split("/"))[1]
    _jaw = splited[2]
    file_name = splited[-1].split(".")[0]
    if _id in result.keys():
        result[_id][f"{_jaw}_{file_name}"] = item
    else:
        result[_id] = {f"{_jaw}_{file_name}": item}

result_df = pd.DataFrame().from_dict(result, orient="index").reset_index(names="id")
result_df.to_csv("formatted_data/teeth3ds.csv", index=False)
