import os
import json
import xml.etree.ElementTree as ET
import pandas as pd
import wandb

# === CONFIGURATION ===
root = '/media/ariels/home2/Git/Grounded-SAM-2/'
LABEL_DIR = os.path.join(root, "datasets/chin_env_data_for_stats/labels/")  # directory containing XML annotation files
PRED_DIR = os.path.join(root, "img_for_stats/car_person_B015_T025/labels/")  # directory containing prediction JSON files
DST_DIR = PRED_DIR[:-7]
EXP_NAME = 'car_person_B015_T025.csv'
IOU_THRESHOLD = 0.3


def parse_voc_xml(xml_path):
    """Parse PASCAL VOC-style XML to extract bounding boxes."""
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
    """Load JSON predictions from a single file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Normalize structure
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
    """Compute Intersection over Union."""
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
    if unionArea == 0:
        return 0.0
    return interArea / unionArea


def evaluate_all_predictions():
    """Evaluate all prediction JSON files in PRED_DIR."""
    all_predictions = {}

    # Load all predictions from all JSON files
    for file in os.listdir(PRED_DIR):
        if not file.endswith(".json"):
            continue
        json_path = os.path.join(PRED_DIR, file)
        preds = load_predictions_from_file(json_path)
        all_predictions.update(preds)

    print(f"Loaded predictions for {len(all_predictions)} images.")

    total_TP = total_FP = total_FN = 0

    # Compare with ground truth
    for filename in os.listdir(LABEL_DIR):
        if not filename.endswith(".xml"):
            continue

        base_name = os.path.splitext(filename)[0]
        xml_path = os.path.join(LABEL_DIR, filename)

        if base_name not in all_predictions:
            print(f"⚠️ Missing predictions for {base_name}")
            continue

        gt_boxes = parse_voc_xml(xml_path)
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

    # === SUMMARY ===
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n=== Detection Evaluation Results (All Files) ===")
    print(f"Total True Positives (TP): {total_TP}")
    print(f"Total False Positives (FP): {total_FP}")
    print(f"Total False Negatives (FN): {total_FN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    columns = ['TP','FP','FN','Precision','Recall','F1']
    df = pd.DataFrame(columns=columns)
    df = pd.DataFrame([{
        'TP': round(total_TP, 4),
        'FP': round(total_FP, 4),
        'FN': round(total_FN, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1': round(f1, 4)
    }])
    df.to_csv(DST_DIR+EXP_NAME, index=False)
    wandb.log({
        "Total_TP": total_TP,
        "Total_FP": total_FP,
        "Total_FN": total_FN,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
    })
# === RUN ===
if __name__ == "__main__":
    wandb.init(
    project="object_detection_eval",  # 🔁 change to your W&B project name
    name="car_person_B015_T025",      # optional experiment name
    config={
        "IOU_THRESHOLD": IOU_THRESHOLD,
        "LABEL_DIR": LABEL_DIR,
        "PRED_DIR": PRED_DIR,
    }
)

    evaluate_all_predictions()

    # --- finish W&B run cleanly ---
    wandb.finish()
