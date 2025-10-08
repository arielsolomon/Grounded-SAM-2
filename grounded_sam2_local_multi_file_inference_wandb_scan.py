

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
import wandb

# ===========================================================
# ✅ WANDB CONFIGURATION
# ===========================================================
os.environ["WANDB_API_KEY"] = "dac846d6e84dafb1a9a54a40976f97adda480161"  # 👈 insert your key
wandb_project = "grounded_sam2_multi_hparam"

# ===========================================================
# Model and data paths
# ===========================================================
IMG_DIR = "datasets/chin_env_data_for_stats/images/"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DUMP_JSON_RESULTS = True

# ===========================================================
# Define hyperparameter grids
# ===========================================================
# TEXT_PROMPTS = [
#     "car",
#     "person.",
#     "person. car.",
#     "car. person. building."
# ]
TEXT_PROMPTS = [
    "car. building."
]

BOX_THRESHOLDS = [0.05, 0.1, 0.15, 0.25, 0.3]
TEXT_THRESHOLDS = [0.05,0.1, 0.15, 0.25, 0.35]

# ===========================================================
# Initialize models once
# ===========================================================
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ===========================================================
# Helper functions
# ===========================================================
def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def clean_prompt_name(prompt: str) -> str:
    """Convert 'car. person.' → 'car_person'"""
    return "_".join(prompt.replace(".", "").replace(",", "").split()).strip("_")

# ===========================================================
# MAIN LOOP
# ===========================================================
image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
image_paths = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if Path(f).suffix.lower() in image_extensions]
print(f"Found {len(image_paths)} images in {IMG_DIR}")

for text_prompt in TEXT_PROMPTS:
    prompt_name = clean_prompt_name(text_prompt)

    for box_th in BOX_THRESHOLDS:
        for text_th in TEXT_THRESHOLDS:
            exp_name = f"{prompt_name}_B{int(box_th*100):03d}_T{int(text_th*100):03d}"
            print(f"\n🚀 Running experiment: {exp_name}")

            # Initialize wandb run
            wandb.init(
                project=wandb_project,
                name=exp_name,
                config={
                    "TEXT_PROMPT": text_prompt,
                    "BOX_THRESHOLD": box_th,
                    "TEXT_THRESHOLD": text_th
                }
            )

            OUTPUT_DIR = Path(f"img_for_stats/{exp_name}")
            images_dir = OUTPUT_DIR / "images"
            labels_dir = OUTPUT_DIR / "labels"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            processed_count = 0
            error_count = 0

            for img_path in tqdm(image_paths, desc=f"{exp_name}"):
                IMG_NAME = Path(img_path).stem

                try:
                    image_source, image = load_image(img_path)
                    sam2_predictor.set_image(image_source)

                    boxes, confidences, labels = predict(
                        model=grounding_model,
                        image=image,
                        caption=text_prompt,
                        box_threshold=box_th,
                        text_threshold=text_th,
                        device=DEVICE
                    )

                    if boxes is None or len(boxes) == 0:
                        continue

                    h, w, _ = image_source.shape
                    boxes = boxes * torch.Tensor([w, h, w, h])
                    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

                    masks, scores, logits = sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_boxes,
                        multimask_output=False,
                    )

                    if masks.ndim == 4:
                        masks = masks.squeeze(1)

                    confidences = confidences.numpy().tolist()
                    class_names = labels
                    class_ids = np.arange(len(class_names))
                    labels_text = [f"{cls} {conf:.2f}" for cls, conf in zip(class_names, confidences)]

                    img = cv2.imread(img_path)
                    detections = sv.Detections(xyxy=input_boxes, mask=masks.astype(bool), class_id=class_ids)
                    annotated = sv.BoxAnnotator().annotate(scene=img.copy(), detections=detections)
                    annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=detections, labels=labels_text)

                    cv2.imwrite(str(images_dir / f"{IMG_NAME}_annotated.jpg"), annotated)

                    if DUMP_JSON_RESULTS:
                        mask_rles = [single_mask_to_rle(mask) for mask in masks]
                        results = {
                            "image_path": img_path,
                            "annotations": [
                                {
                                    "class_name": c,
                                    "bbox": b.tolist(),
                                    "segmentation": m,
                                    "score": s
                                }
                                for c, b, m, s in zip(class_names, input_boxes, mask_rles, scores.tolist())
                            ]
                        }
                        with open(labels_dir / f"{IMG_NAME}_results.json", "w") as f:
                            json.dump(results, f, indent=4)

                    processed_count += 1

                except Exception as e:
                    print(f"❌ Error processing {img_path}: {e}")
                    error_count += 1

            # Log experiment stats
            wandb.log({
                "processed_images": processed_count,
                "errors": error_count
            })
            wandb.finish()

print("\n✅ All experiments completed.")
