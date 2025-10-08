import os
import json
import xml.etree.ElementTree as ET
import pandas as pd
import wandb

# === CONFIGURATION ===
os.environ["WANDB_API_KEY"] = "dac846d6e84dafb1a9a54a40976f97adda480161" 
root = '/work/Grounded-SAM-2/'
BASE_PRED_ROOT = os.path.join(root, "img_for_stats/")  # folder containing multiple experiment folders
LABEL_DIR = os.path.join(root, "datasets/chin_env_data_for_stats/labels/")  # ground truth labels
IOU_THRESHOLD = 0.3

# === FUNCTIONS ===
def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes


def load_predictions_from_file(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "annotations" in data:
        preds = {os.path.basename(data["image_path"]).split(".")[0]: data["annotations"]}
    elif isinstance(data, list):
        preds = {}
        for item in data:
            base = os.path.basename(item["image_path"]).split(".")[0]
            preds[base] = item["annotations"]
    else:
        raise ValueError(f"Unsupported JSON format: {json_path}")
    return preds


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0.0


def evaluate_experiment(exp_name, pred_dir):
    all_predictions = {}
    for file in os.listdir(pred_dir):
        if not file.endswith(".json"):
            continue
        preds = load_predictions_from_file(os.path.join(pred_dir, file))
        all_predictions.update(preds)

    total_TP = total_FP = total_FN = 0

    for filename in os.listdir(LABEL_DIR):
        if not filename.endswith(".xml"):
            continue
        base_name = os.path.splitext(filename)[0]
        if base_name not in all_predictions:
            continue

        gt_boxes = parse_voc_xml(os.path.join(LABEL_DIR, filename))
        pred_boxes = [ann["bbox"] for ann in all_predictions[base_name]]

        matched_gt = set()
        TP = FP = 0
        for pb in pred_boxes:
            found_match = False
            for i, gb in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                if iou(pb, gb) >= IOU_THRESHOLD:
                    TP += 1
                    matched_gt.add(i)
                    found_match = True
                    break
            if not found_match:
                FP += 1

        FN = len(gt_boxes) - len(matched_gt)
        total_TP += TP
        total_FP += FP
        total_FN += FN

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        'Experiment': exp_name,
        'TP': total_TP,
        'FP': total_FP,
        'FN': total_FN,
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1': round(f1, 4)
    }

    df = pd.DataFrame([results])
    df.to_csv(os.path.join(pred_dir, f"{exp_name}_results.csv"), index=False)

    wandb.log(results)
    return results


# === MAIN ===
if __name__ == "__main__":
    all_results = []
    wandb_project = "G_DINO_OB_EVAL"

    for exp_name in sorted(os.listdir(BASE_PRED_ROOT)):
        exp_path = os.path.join(BASE_PRED_ROOT, exp_name)
        if not os.path.isdir(exp_path):
            continue
        label_subdir = os.path.join(exp_path, "labels")
        if not os.path.exists(label_subdir):
            print(f"⚠️ Skipping {exp_name}: no 'labels' folder found.")
            continue

        print(f"\n🚀 Running evaluation for: {exp_name}")
        wandb.init(project=wandb_project, name=exp_name, config={
            "IOU_THRESHOLD": IOU_THRESHOLD,
            "LABEL_DIR": LABEL_DIR,
            "PRED_DIR": label_subdir
        })

        res = evaluate_experiment(exp_name, label_subdir)
        all_results.append(res)

        wandb.finish()

    if all_results:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(os.path.join(BASE_PRED_ROOT, "all_experiments_results.csv"), index=False)
        print("\n✅ All experiments completed. Combined results saved to 'all_experiments_results.csv'.")