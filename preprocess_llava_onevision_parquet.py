import os
from datasets import load_dataset
from tqdm import tqdm
import json

data = load_dataset("lmms-lab/LLaVA-OneVision-Data",'FigureQA(MathV360K)', split="train")

image_folder = "./mydatasets/llava_onevision/images/FigureQA_MathV360K/"

converted_data = []

for da in tqdm(data):
    json_data = {}
    json_data["id"] = da["id"]
    if da["image"] is not None:
        json_data["image"] = f"{da['id']}.rgba"
        da["image"].save(os.path.join(image_folder, json_data["image"]))
    json_data["conversations"] = da["conversations"]
    converted_data.append(json_data)


with open("./mydatasets/llava_onevision/llava_onevision_FigureQA_MathV360K.json", "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)
