import os
from datasets import load_dataset
from tqdm import tqdm
import json

# Load the dataset
dataset = load_dataset("laolao77/ViRFT_CLS_flower_4_shot")

# Create output directory if it doesn't exist
image_folder = "./msdata/ViRFT_CLS_flower_4_shot/images/"
os.makedirs(image_folder, exist_ok=True)

converted_data = []

# Process each split in the dataset
for split in dataset:
    for idx, item in enumerate(tqdm(dataset[split])):
        json_data = {}
        # Use index as id since there's no id field
        json_data["id"] = f"{split}_{idx}"
        
        if item["image"] is not None:
            json_data["image"] = f"{json_data['id']}.rgba"
            item["image"].save(os.path.join(image_folder, json_data["image"]))
        
        # Create conversations based on problem and solution
        json_data["conversations"] = [
            {
                "from": "human",
                "value": item["problem"]
            },
            {
                "from": "gpt",
                "value": item["solution"]
            }
        ]
        
        converted_data.append(json_data)

with open("./msdata/ViRFT_CLS_flower_4_shot/images/ViRFT_CLS_flower_4_shot.json", "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)
