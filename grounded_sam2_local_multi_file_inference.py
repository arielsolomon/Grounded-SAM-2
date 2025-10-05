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
from tqdm import tqdm


"""
Hyper parameters
"""
TEXT_PROMPT = "car. "
IMG_DIR = "datasets/chin_env_data_for_stats/images/"  # directory containing multiple images
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.15
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("img_for_stats")
DUMP_JSON_RESULTS = True

# ----------------------------------------
# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Build SAM2 image predictor
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# Build grounding DINO model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

# Autocast setup for efficiency
#torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Function to convert single mask to RLE
def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# Iterate over all images in directory
image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
image_paths = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if Path(f).suffix.lower() in image_extensions]

print(f"Found {len(image_paths)} images in {IMG_DIR}")

for img_path in tqdm(image_paths, desc="running inference"):
    IMG_NAME = Path(img_path).stem
    print(f"\nProcessing: {IMG_NAME}")

    try:
        # Load image
        image_source, image = load_image(img_path)
        sam2_predictor.set_image(image_source)

        # Predict boxes using Grounding DINO
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE
        )

        # Skip if no detections
        if boxes is None or len(boxes) == 0:
            print(f"No objects found in {IMG_NAME}")
            continue

        # Convert boxes to xyxy format and run SAM2 segmentation
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Convert masks to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences = confidences.numpy().tolist()
        class_names = labels
        class_ids = np.arange(len(class_names))

        # Prepare labels for visualization
        labels_text = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(class_names, confidences)
        ]

        # Visualize and save annotated image
        img = cv2.imread(img_path)
        detections = sv.Detections(xyxy=input_boxes, mask=masks.astype(bool), class_id=class_ids)

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated = box_annotator.annotate(scene=img.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels_text)

        out_img_path = OUTPUT_DIR / "images" / f"{IMG_NAME}_annotated.jpg"
        cv2.imwrite(str(out_img_path), annotated)

        # Dump JSON results
        if DUMP_JSON_RESULTS:
            mask_rles = [single_mask_to_rle(mask) for mask in masks]
            input_boxes_list = input_boxes.tolist()
            scores_list = scores.tolist()

            results = {
                "image_path": img_path,
                "annotations": [
                    {
                        "class_name": cname,
                        "bbox": box,
                        "segmentation": mask_rle,
                        "score": score,
                    }
                    for cname, box, mask_rle, score in zip(class_names, input_boxes_list, mask_rles, scores_list)
                ],
                "box_format": "xyxy",
                "img_width": w,
                "img_height": h,
            }

            out_json_path = OUTPUT_DIR / "labels" / f"{IMG_NAME}_results.json"
            with open(out_json_path, "w") as f:
                json.dump(results, f, indent=4)

        print(f"Saved annotated image and JSON for {IMG_NAME}")

    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")

print("\n✅ Done! All images processed.")
