import os
from datasets import load_dataset
from tqdm import tqdm
import json
import shutil
import base64
from io import BytesIO
from PIL import Image

data = load_dataset("Fancy-MLLM/R1-Onevision",'default', split="train")

image_folder = "./msdata/onevision_r1/images/VizWiz_MathV360K/"
os.makedirs(image_folder, exist_ok=True)

converted_data = []

for da in tqdm(data):
    json_data = {}
    json_data["id"] = da["id"]
    if da["image"] is not None:
        json_data["image"] = f"{da['id']}.rgba"
        # Handle base64 encoded image
        if isinstance(da["image"], str):
            try:
                # Try to decode base64 string
                if "base64," in da["image"]:
                    # Remove the data URL prefix if present
                    base64_data = da["image"].split("base64,")[1]
                else:
                    base64_data = da["image"]
                
                image_data = base64.b64decode(base64_data)
                image = Image.open(BytesIO(image_data))
                image.save(os.path.join(image_folder, json_data["image"]))
            except Exception as e:
                print(f"Error processing image for id {da['id']}: {str(e)}")
                continue
        else:
            # If it's a PIL Image (though this case seems unlikely now)
            da["image"].save(os.path.join(image_folder, json_data["image"]))
    json_data["conversations"] = da["conversations"]
    converted_data.append(json_data)

# Create parent directory if it doesn't exist
os.makedirs(os.path.dirname("./msdata/onevision_r1/VizWiz_MathV360K/onevision_r1_VizWiz_MathV360K.json"), exist_ok=True)

with open("./msdata/onevision_r1/VizWiz_MathV360K/onevision_r1_VizWiz_MathV360K.json", "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)
