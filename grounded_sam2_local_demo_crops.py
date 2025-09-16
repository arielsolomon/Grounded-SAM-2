import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from torchvision.ops import box_convert
import supervision as sv
import pycocotools.mask as mask_util

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

# -----------------------
# Hyperparameters / paths
# -----------------------
TEXT_PROMPT = "car. person."
IMG_PATH = "notebooks/images/truck.jpg"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs")
CROP_SIZE = 4000
DUMP_JSON_RESULTS = True

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Build SAM2 predictor
# -----------------------
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# -----------------------
# Build Grounding DINO model
# -----------------------
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

# -----------------------
# CropDetector Class
# -----------------------
class CropDetector:
    def __init__(
        self,
        grounding_model,
        sam2_predictor,
        class_name="car",
        box_threshold=0.35,
        text_threshold=0.25,
        device="cuda",
        crop_size=4000,
        output_dir=Path("outputs/grounded_sam2_local_demo"),
    ):
        self.grounding_model = grounding_model
        self.sam2_predictor = sam2_predictor
        self.class_name = class_name
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.crop_size = crop_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def single_mask_to_rle(self, mask):
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    def run_on_image(self, img_path, text_prompt):
        # load image
        orig_image_bgr = cv2.imread(img_path)
        h_img, w_img = orig_image_bgr.shape[:2]
        detections_yolo = []

        x_steps = list(range(0, w_img, self.crop_size))
        y_steps = list(range(0, h_img, self.crop_size))

        # Merge results for optional JSON / visualization
        all_boxes = []
        all_masks = []
        all_confidences = []
        all_labels = []

        for x0 in x_steps:
            for y0 in y_steps:
                x1 = min(x0 + self.crop_size, w_img)
                y1 = min(y0 + self.crop_size, h_img)
                crop = orig_image_bgr[y0:y1, x0:x1]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                self.sam2_predictor.set_image(crop_rgb)

                # Grounding DINO detection
                boxes, confidences, labels = predict(
                    model=self.grounding_model,
                    image=crop_rgb,
                    caption=text_prompt,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    device=self.device
                )

                if len(boxes) == 0:
                    continue

                # Convert boxes to pixels
                h_crop, w_crop, _ = crop_rgb.shape
                boxes = boxes * torch.Tensor([w_crop, h_crop, w_crop, h_crop])
                input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

                # Run SAM2 predictor
                masks, scores, _ = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
                if masks.ndim == 4:
                    masks = masks.squeeze(1)

                # Save for visualization / JSON
                all_boxes.extend(input_boxes + np.array([x0, y0, x0, y0]))
                all_masks.extend(masks)
                all_confidences.extend(confidences.numpy().tolist())
                all_labels.extend(labels)

                # Convert boxes to YOLO format
                for box in input_boxes:
                    x_min, y_min, x_max, y_max = box
                    x_min += x0
                    x_max += x0
                    y_min += y0
                    y_max += y0

                    x_center = (x_min + x_max) / 2 / w_img
                    y_center = (y_min + y_max) / 2 / h_img
                    width = (x_max - x_min) / w_img
                    height = (y_max - y_min) / h_img

                    detections_yolo.append([0, x_center, y_center, width, height])

        # Save YOLO labels
        yolo_txt_path = self.output_dir / (Path(img_path).stem + ".txt")
        with open(yolo_txt_path, "w") as f:
            for det in detections_yolo:
                f.write(" ".join([str(x) for x in det]) + "\n")
        print(f"Saved YOLO labels to {yolo_txt_path}")

        # Optional JSON output
        if DUMP_JSON_RESULTS and len(all_boxes) > 0:
            mask_rles = [self.single_mask_to_rle(mask) for mask in all_masks]
            results = {
                "image_path": img_path,
                "annotations" : [
                    {
                        "class_name": cname,
                        "bbox": box.tolist(),
                        "segmentation": mask_rle,
                        "score": score,
                    }
                    for cname, box, mask_rle, score in zip(all_labels, all_boxes, mask_rles, all_confidences)
                ],
                "box_format": "xyxy",
                "img_width": w_img,
                "img_height": h_img,
            }
            json_path = self.output_dir / (Path(img_path).stem + "_results.json")
            with open(json_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Saved JSON results to {json_path}")

        # -----------------------
        # Visualization on original image
        # -----------------------
        if len(all_boxes) > 0:
            img_vis = orig_image_bgr.copy()
            all_boxes_arr = np.array(all_boxes)
            all_masks_arr = np.array(all_masks)
            class_ids = np.arange(len(all_labels))
            labels_vis = [f"{c} {s:.2f}" for c, s in zip(all_labels, all_confidences)]

            detections = sv.Detections(
                xyxy=all_boxes_arr,
                mask=all_masks_arr.astype(bool),
                class_id=class_ids
            )
            box_annotator = sv.BoxAnnotator()
            img_vis = box_annotator.annotate(img_vis, detections)
            label_annotator = sv.LabelAnnotator()
            img_vis = label_annotator.annotate(img_vis, detections, labels_vis)
            mask_annotator = sv.MaskAnnotator()
            img_vis = mask_annotator.annotate(img_vis, detections)

            vis_path = self.output_dir / (Path(img_path).stem + "trial1.jpg")
            cv2.imwrite(vis_path, img_vis)
            print(f"Saved visualized image with all detections to {vis_path}")

        return detections_yolo

# -----------------------
# Run detection on image
# -----------------------
detector = CropDetector(
    grounding_model=grounding_model,
    sam2_predictor=sam2_predictor,
    class_name="car",
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    device=DEVICE,
    crop_size=CROP_SIZE,
    output_dir=OUTPUT_DIR
)

detections = detector.run_on_image(IMG_PATH, TEXT_PROMPT)
