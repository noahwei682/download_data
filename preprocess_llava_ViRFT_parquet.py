import os
from datasets import load_dataset
from tqdm import tqdm
import json

data = load_dataset("laolao77/ViRFT_CLS_flower_4_shot")

image_folder = "./msdata/ViRFT_CLS_flower_4_shot/images/"

converted_data = []

for da in tqdm(data):
    json_data = {}
    json_data["id"] = da["id"]
    if da["image"] is not None:
        json_data["image"] = f"{da['id']}.rgba"
        da["image"].save(os.path.join(image_folder, json_data["image"]))
    json_data["conversations"] = da["conversations"]
    converted_data.append(json_data)


with open("./msdata/ViRFT_CLS_flower_4_shot/images/ViRFT_CLS_flower_4_shot.json", "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)
