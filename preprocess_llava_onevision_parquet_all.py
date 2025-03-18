import os
from datasets import load_dataset
from tqdm import tqdm
import json

datasets = ['CLEVR-Math(MathV360K)', 'FigureQA(MathV360K)', 'GEOS(MathV360K)', 'GeoQA+(MathV360K)', 'Geometry3K(MathV360K)', 'IconQA(MathV360K)', 'MapQA(MathV360K)', 'PMC-VQA(MathV360K)', 'Super-CLEVR(MathV360K)', 'TabMWP(MathV360K)', 'UniGeo(MathV360K)', 'VisualWebInstruct(filtered)', 'VizWiz(MathV360K)', 'ai2d(cauldron,llava_format)', 'ai2d(gpt4v)', 'ai2d(internvl)', 'allava_instruct_laion4v', 'allava_instruct_vflan4v', 'aokvqa(cauldron,llava_format)', 'chart2text(cauldron)', 'chartqa(cauldron,llava_format)', 'chrome_writting', 'clevr(cauldron,llava_format)', 'diagram_image_to_text(cauldron)', 'dvqa(cauldron,llava_format)', 'figureqa(cauldron,llava_format)', 'geo170k(align)', 'geo170k(qa)', 'geo3k', 'geomverse(cauldron)', 'hateful_memes(cauldron,llava_format)', 'hitab(cauldron,llava_format)', 'hme100k', 'iam(cauldron)', 'iconqa(cauldron,llava_format)', 'iiit5k', 'image_textualization(filtered)', 'infographic(gpt4v)', 'infographic_vqa', 'infographic_vqa_llava_format', 'intergps(cauldron,llava_format)', 'k12_printing', 'llavar_gpt4_20k', 'lrv_chart', 'lrv_normal(filtered)', 'magpie_pro(l3_80b_mt)', 'magpie_pro(l3_80b_st)', 'magpie_pro(qwen2_72b_st)', 'mapqa(cauldron,llava_format)', 'mathqa', 'mavis_math_metagen', 'mavis_math_rule_geo', 'multihiertt(cauldron)', 'orand_car_a', 'raven(cauldron)', 'rendered_text(cauldron)', 'robut_sqa(cauldron)', 'robut_wikisql(cauldron)', 'robut_wtq(cauldron,llava_format)', 'scienceqa(cauldron,llava_format)', 'scienceqa(nona_context)', 'screen2words(cauldron)', 'sharegpt4o', 'sharegpt4v(coco)', 'sharegpt4v(knowledge)', 'sharegpt4v(llava)', 'sharegpt4v(sam)', 'sroie', 'st_vqa(cauldron,llava_format)', 'tabmwp(cauldron)', 'tallyqa(cauldron,llava_format)', 'textcaps', 'textocr(gpt4v)', 'tqa(cauldron,llava_format)', 'ureader_cap', 'ureader_ie', 'vision_flan(filtered)', 'vistext(cauldron)', 'visual7w(cauldron,llava_format)', 'visualmrc(cauldron)', 'vqarad(cauldron,llava_format)', 'vsr(cauldron,llava_format)', 'websight(cauldron)']

base_dir = "./mydatasets/llava_onevision"
os.makedirs(base_dir, exist_ok=True)

yaml_config = {"datasets": []}

for dataset in datasets:
    try:
        print(f"Processing dataset: {dataset}")
        
        data = load_dataset("lmms-lab/LLaVA-OneVision-Data", dataset, split="train")
        
  
        converted_string = dataset.replace("(", "_").replace(")", "").replace(",", "_").replace("+", "Plus")
        
        image_folder = os.path.join(base_dir, converted_string)
        os.makedirs(image_folder, exist_ok=True)
        
        output_json_path = os.path.join(base_dir, f"llava_onevision_{converted_string}.json")
        
        converted_data = []
        
        for da in tqdm(data, desc=f"Processing samples in {dataset}"):
            json_data = {}
            json_data["id"] = da["id"]
            if da["image"] is not None:
                json_data["image"] = f"{da['id']}.rgba"
                da["image"].save(os.path.join(image_folder, json_data["image"]))
            json_data["conversations"] = da["conversations"]
            converted_data.append(json_data)
        
        with open(output_json_path, "w") as f:
            json.dump(converted_data, f, indent=4, ensure_ascii=False)


        yaml_config["datasets"].append({
            "json_path": output_json_path,
            "sampling_strategy": "all"
        })
        
        print(f"Completed processing {dataset}: {len(converted_data)} samples")
        
    except Exception as e:
        print(f"Error processing dataset {dataset}: {str(e)}")
        continue

import yaml
with open(os.path.join(base_dir, "llava_onevision_all_datasets.yaml"), "w") as f:
    yaml.dump(yaml_config, f, default_flow_style=False)

print("All datasets processed. YAML config file created for multi-dataset training.")
