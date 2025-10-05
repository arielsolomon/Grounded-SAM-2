import os
import json
import xml.etree.ElementTree as ET

# === CONFIGURATION ===
LABEL_DIR = "/mnt/data"  # directory containing XML annotation files
PRED_FILE = "/mnt/data/grounded_sam2_local_image_demo_results.json"  # prediction results
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


def load_predictions(json_path):
    """Load JSON predictions — handle single-image or multi-image formats."""
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "annotations" in data:
        # single image case
        preds = {os.path.basename(data["image_path"]).split(".")[0]: data["annotations"]}
    elif isinstance(data, list):
        # multiple images in a list
        preds = {}
        for item in data:
            base = os.path.basename(item["image_path"]).split(".")[0]
            preds[base] = item["annotations"]
    else:
        raise ValueError("Unsupported JSON format for predictions.")
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


def evaluate_detections():
    predictions = load_predictions(PRED_FILE)

    total_TP = total_FP = total_FN = 0

    for filename in os.listdir(LABEL_DIR):
        if not filename.endswith(".xml"):
            continue

        base_name = os.path.splitext(filename)[0]
        xml_path = os.path.join(LABEL_DIR, filename)

        if base_name not in predictions:
            print(f"⚠️ Missing predictions for {base_name}")
            continue

        gt_boxes = parse_voc_xml(xml_path)
        pred_boxes = [ann["bbox"] for ann in predictions[base_name]]

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

    print("\n=== Detection Evaluation Results ===")
    print(f"True Positives (TP): {total_TP}")
    print(f"False Positives (FP): {total_FP}")
    print(f"False Negatives (FN): {total_FN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


# === RUN ===
if __name__ == "__main__":
    evaluate_detections()
