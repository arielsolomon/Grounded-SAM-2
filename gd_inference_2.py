import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

torch.cuda.empty_cache() 

"""
Hyper parameters
"""
TEXT_PROMPT = "car bus building"
IMG_PATH = "/work/data/china_env_dataset/mixed_images/images/4_148.bmp"
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs")
DUMP_JSON_RESULTS = True
DUMP_YOLOV11_LABELS = True
YOLO_LABEL_DIR = OUTPUT_DIR / "yolov11_labels"

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
YOLO_LABEL_DIR.mkdir(parents=True, exist_ok=True)

# build SAM2 image predictor
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build GroundingDINO model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

# Load image
image_source, image = load_image(IMG_PATH)
sam2_predictor.set_image(image_source)
h, w, _ = image_source.shape

# Split prompts
prompts = [p.strip() for p in TEXT_PROMPT.split() if p.strip()]

all_boxes = []
all_confidences = []
all_labels = []
all_masks = []

for text in prompts:
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )

    if len(boxes) == 0:
        continue

    # Convert boxes for SAM
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # SAM prediction
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    all_boxes.append(input_boxes)
    all_confidences.extend(confidences.numpy().tolist())
    all_labels.extend(labels)
    all_masks.append(masks)

# Combine all results
if all_boxes:
    all_boxes = np.vstack(all_boxes)
    all_masks = np.vstack(all_masks)
else:
    all_boxes = np.zeros((0, 4))
    all_masks = np.zeros((0, h, w))

class_names = all_labels
confidences = all_confidences
class_ids = np.arange(len(class_names))

labels_text = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence in zip(class_names, confidences)
]

# Visualization
img = cv2.imread(IMG_PATH)
detections = sv.Detections(
    xyxy=all_boxes,
    mask=all_masks.astype(bool),
    class_id=class_ids
)

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

label_annotator = sv.LabelAnnotator()
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels_text)

mask_annotator = sv.MaskAnnotator()
annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image.jpg"), annotated_frame)

# Function to encode mask in RLE
def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

# Save JSON results
if DUMP_JSON_RESULTS:
    mask_rles = [single_mask_to_rle(mask) for mask in all_masks]
    results = {
        "image_path": IMG_PATH,
        "annotations": [
            {
                "class_name": class_name,
                "bbox": box.tolist(),
                "segmentation": mask_rle,
                "score": score,
            }
            for class_name, box, mask_rle, score in zip(class_names, all_boxes, mask_rles, confidences)
        ],
        "box_format": "xyxy",
        "img_width": w,
        "img_height": h
    }
    with open(os.path.join(OUTPUT_DIR, "grounded_sam2_results.json"), "w") as f:
        json.dump(results, f, indent=4)

# Save YOLOv11 labels
if DUMP_YOLOV11_LABELS:
    for class_name, box in zip(class_names, all_boxes):
        x1, y1, x2, y2 = box
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h
        class_id = 0  # You can map multiple classes to IDs here if needed

        txt_filename = os.path.splitext(os.path.basename(IMG_PATH))[0] + ".txt"
        with open(YOLO_LABEL_DIR / txt_filename, "a") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("Processing complete. Outputs saved to:", OUTPUT_DIR)