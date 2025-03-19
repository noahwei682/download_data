import os
from datasets import load_dataset
from tqdm import tqdm
import json

data = load_dataset("wei682/Cambrian-LLaVA-Amazon-MLLM-objects",'All_Beauty', split="train")

image_folder = "./mydatasets/amazon/images/all_beauty/"

converted_data = []

for da in tqdm(data):
    json_data = {}
    json_data["id"] = da["id"]
    if da["image"] is not None:
        json_data["image"] = f"{da['id']}.jpg"
        da["image"].save(os.path.join(image_folder, json_data["image"])) 
    json_data["conversations"] = da["conversations"]
    converted_data.append(json_data)


with open("./mydatasets/amazon/amazon_all_beauty_qa_it_data.json", "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)
