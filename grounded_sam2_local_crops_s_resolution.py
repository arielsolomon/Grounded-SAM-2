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
from pathlib import Path
import math
import tempfile
from PIL import Image
from tqdm import tqdm
from time import time
import torchvision.transforms as transforms
from torchsr.models import edsr_baseline


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", category=UserWarning)


start = time()
"""
Hyper parameters
"""

TEXT_PROMPT = "car. person."
IMG_PATH = "datasets/56_029.bmp"
IMG_NAME = Path(IMG_PATH).stem 
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs")
DUMP_JSON_RESULTS = True

# Tiling parameters
USE_TILING = True # <--- New Flag: Set to False to process the entire image at once
CROP_SIZE = 150  # Size of each crop (single dimension - creates 150x150 crops)
OVERLAP = 15     # Overlap between crops to avoid missing objects at edges
SAVE_CROPS = False # Set to False if you don't want to save crop images

# Super Resolution parameters
USE_SUPER_RESOLUTION = True 
SR_SCALE_FACTOR = 4  # Scale factor for super resolution (2x, 3x, 4x)
SR_MODEL = "edsr_x4"  # TorchSR model name
SR_CROP_SIZE = "4x_"
SAVE_SR_CROPS = True  # Save super resolution crops for inspection

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# create crops subdirectory if saving crops
CROPS_DIR = OUTPUT_DIR / "crops"
if SAVE_CROPS and USE_TILING:
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

# create super resolution crops subdirectory
SR_CROPS_DIR = OUTPUT_DIR / f"sr_crops_{SR_CROP_SIZE[:-1]}"
print(f"\nSR crops dir: {SR_CROPS_DIR}\n")
if SAVE_SR_CROPS and USE_SUPER_RESOLUTION and USE_TILING:
    SR_CROPS_DIR.mkdir(parents=True, exist_ok=True)

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

# Initialize super resolution model
sr_model = None
if USE_SUPER_RESOLUTION:
    try:
        print(f"Loading super resolution model: {SR_MODEL}")
        sr_model = edsr_baseline(scale=4, pretrained=True).to(DEVICE)
        sr_model.eval()
        print(f"\nSuper resolution model loaded successfully\n")
    except Exception as e:
        print(f"Error loading super resolution model: {e}")
        USE_SUPER_RESOLUTION = False
        print("Continuing without super resolution")

# setup the input image and text prompt for SAM 2 and Grounding DINO
text = TEXT_PROMPT
img_path = IMG_PATH

# Load the original image using OpenCV to get dimensions
original_cv_image = cv2.imread(img_path)
orig_h, orig_w, _ = original_cv_image.shape

print(f"Original image size: {orig_w} x {orig_h}")

def generate_crops(image_width, image_height, crop_size, overlap):
    """Generate crop coordinates with overlap"""
    crops = []
    
    # Calculate step size (crop_size - overlap)
    step = crop_size - overlap
    
    # Generate crops
    for y in range(0, image_height, step):
        for x in range(0, image_width, step):
            # Calculate crop boundaries
            x1 = x
            y1 = y
            x2 = min(x + crop_size, image_width)
            y2 = min(y + crop_size, image_height)
            
            # Skip if crop is too small
            if (x2 - x1) < crop_size // 2 or (y2 - y1) < crop_size // 2:
                continue
                
            crops.append((x1, y1, x2, y2))
    
    return crops

def apply_super_resolution(image, sr_model, device):
    """Apply super resolution to an image using torchsr"""
    if sr_model is None:
        return image
    
    try:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        tensor_image = transform(pil_image).unsqueeze(0).to(device)
        
        # Apply super resolution
        with torch.no_grad():
            sr_tensor = sr_model(tensor_image)
        
        # Convert back to numpy array
        sr_tensor = sr_tensor.squeeze(0).cpu()
        sr_tensor = torch.clamp(sr_tensor, 0, 1)
        
        # Convert to numpy and back to BGR
        sr_image = sr_tensor.permute(1, 2, 0).numpy()
        sr_image = (sr_image * 255).astype(np.uint8)
        sr_image_bgr = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        
        return sr_image_bgr
        
    except Exception as e:
        print(f"Error in super resolution: {e}")
        return image

def convert_to_yolo_format(bbox, img_width, img_height):
    """Convert xyxy bbox to YOLO format (xcent, ycent, w, h) normalized"""
    x1, y1, x2, y2 = bbox
    
    # Calculate center and dimensions
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    
    # Normalize to image dimensions
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm

def non_max_suppression_custom(detections, iou_threshold=0.5):
    """Custom NMS to remove overlapping detections from different crops"""
    if len(detections) == 0:
        return []
    
    # Sort by confidence score (descending)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    
    for i, det in enumerate(detections):
        bbox1 = det['bbox']
        keep_detection = True
        
        for kept_det in keep:
            bbox2 = kept_det['bbox']
            
            # Calculate IoU
            x1 = max(bbox1[0], bbox2[0])
            y1 = max(bbox1[1], bbox2[1])
            x2 = min(bbox1[2], bbox2[2])
            y2 = min(bbox1[3], bbox2[3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                union = area1 + area2 - intersection
                
                iou = intersection / union if union > 0 else 0
                
                if iou > iou_threshold:
                    keep_detection = False
                    break
        
        if keep_detection:
            keep.append(det)
    
    return keep

# Setup autocast for SAM2 only (not for Grounding DINO)
use_autocast = torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8

if use_autocast:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Generate crop coordinates based on the USE_TILING flag
if USE_TILING:
    crops = generate_crops(orig_w, orig_h, CROP_SIZE, OVERLAP)
    print(f"\nGenerated {len(crops)} crops\n")
else:
    # Process the entire image as a single "crop"
    crops = [(0, 0, orig_w, orig_h)]
    print("\nProcessing the entire image without tiling\n")

# Store all detections
all_detections = []
all_yolo_labels = []

# Create temporary directory for crop images
temp_dir = tempfile.mkdtemp()

for crop_idx, (x1, y1, x2, y2) in tqdm(enumerate(crops), total=len(crops), desc="Processing images"):
    # Extract crop from original image
    crop_image = original_cv_image[y1:y2, x1:x2]
    crop_h, crop_w, _ = crop_image.shape
    
    # Save original crop to crops directory if enabled and tiling is used
    if SAVE_CROPS and USE_TILING:
        crop_filename = f"crop_{crop_idx:03d}_x{x1}-{x2}_y{y1}-{y2}.jpg"
        crop_save_path = CROPS_DIR / crop_filename
        cv2.imwrite(str(crop_save_path), crop_image)
    
    # Apply super resolution to the crop
    processed_crop = crop_image
    if USE_SUPER_RESOLUTION and sr_model is not None:
        processed_crop = apply_super_resolution(crop_image, sr_model, DEVICE)
        
        # Save super resolution crop if enabled and tiling is used
        if SAVE_SR_CROPS and USE_TILING:
            sr_crop_filename = f"sr_crop_{SR_CROP_SIZE}{crop_idx:03d}_x{x1}-{x2}_y{y1}-{y2}.jpg"
            sr_crop_save_path = SR_CROPS_DIR / sr_crop_filename
            print(f'\n crop file path: {str(sr_crop_save_path)}\n')
            cv2.imwrite(str(sr_crop_save_path), processed_crop)
    
    # Get dimensions of processed crop (might be different due to super resolution)
    processed_h, processed_w, _ = processed_crop.shape
    
    # Save processed crop as temporary image file for grounding dino
    temp_crop_path = os.path.join(temp_dir, f"crop_{crop_idx}_{SR_CROP_SIZE[:-1]}.jpg")
    cv2.imwrite(temp_crop_path, processed_crop)
    
    try:
        # Load processed crop using grounding_dino's load_image function
        crop_image_source, crop_image_pil = load_image(temp_crop_path)
        
        # Set image for SAM2 predictor
        sam2_predictor.set_image(crop_image_source)
        
        # Predict on processed crop (without autocast for Grounding DINO)
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=crop_image_pil,
            caption=text,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE
        )
        
        if len(boxes) > 0:
            # Process boxes for SAM2 - use processed crop dimensions
            boxes = boxes * torch.Tensor([processed_w, processed_h, processed_w, processed_h])
            input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            
            # Get masks from SAM2 (with autocast only for SAM2)
            if use_autocast:
                with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                    masks, scores, logits = sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_boxes,
                        multimask_output=False,
                    )
            else:
                masks, scores, logits = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
            
            # Convert masks shape if needed
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            
            # Convert crop coordinates to original image coordinates
            for i, (box, confidence, label, mask, score) in enumerate(zip(input_boxes, confidences, labels, masks, scores)):
                # Scale back coordinates if super resolution was applied
                if USE_SUPER_RESOLUTION and sr_model is not None:
                    # Scale down the coordinates by the super resolution factor
                    scale_factor = SR_SCALE_FACTOR
                    box = box / scale_factor
                
                # Convert box coordinates to original image space
                orig_x1 = box[0] + x1
                orig_y1 = box[1] + y1
                orig_x2 = box[2] + x1
                orig_y2 = box[3] + y1
                
                orig_bbox = [orig_x1, orig_y1, orig_x2, orig_y2]
                
                # Store detection
                detection = {
                    'bbox': orig_bbox,
                    'confidence': float(confidence),
                    'class_name': label,
                    'score': float(score.max()) if isinstance(score, np.ndarray) else float(score),
                    'mask': mask,
                    'crop_offset': (x1, y1),
                    'used_super_resolution': USE_SUPER_RESOLUTION and sr_model is not None
                }
                all_detections.append(detection)
                
                # Convert to YOLO format
                yolo_coords = convert_to_yolo_format(orig_bbox, orig_w, orig_h)
                yolo_label = f"0 {yolo_coords[0]:.6f} {yolo_coords[1]:.6f} {yolo_coords[2]:.6f} {yolo_coords[3]:.6f}"
                all_yolo_labels.append(yolo_label)
        
        # Clean up temporary crop file
        os.remove(temp_crop_path)
    
    except Exception as e:
        tqdm.write(f"Error processing crop {crop_idx}: {e}")
        # Clean up temporary crop file even on error
        if os.path.exists(temp_crop_path):
            os.remove(temp_crop_path)
        continue

# Clean up temporary directory
try:
    os.rmdir(temp_dir)
except:
    pass

print(f"Total detections before NMS: {len(all_detections)}")

# Apply Non-Maximum Suppression to remove duplicates
if USE_TILING:
    # Only apply NMS if tiling was used
    filtered_detections = non_max_suppression_custom(all_detections, iou_threshold=0.5)
else:
    # No NMS needed for a single image pass
    filtered_detections = all_detections

print(f"Total detections after NMS: {len(filtered_detections)}")

# Save YOLO format labels (after NMS)
yolo_labels_filtered = []
for det in filtered_detections:
    yolo_coords = convert_to_yolo_format(det['bbox'], orig_w, orig_h)
    yolo_label = f"0 {yolo_coords[0]:.6f} {yolo_coords[1]:.6f} {yolo_coords[2]:.6f} {yolo_coords[3]:.6f}"
    yolo_labels_filtered.append(yolo_label)

# Write YOLO labels to file
with open(os.path.join(OUTPUT_DIR, f"{IMG_NAME}_sr_{SR_CROP_SIZE}bx_thr{BOX_THRESHOLD}_txt_thr{TEXT_THRESHOLD}.txt"), "w") as f:
    for label in yolo_labels_filtered:
        f.write(label + "\n")

print(f"Saved YOLO labels to {os.path.join(OUTPUT_DIR, f'{IMG_NAME}_sr_{SR_CROP_SIZE}.txt')}")

# Visualize all detections on original image
if filtered_detections:
    # Prepare data for supervision
    xyxy_boxes = np.array([det['bbox'] for det in filtered_detections])
    confidences = np.array([det['confidence'] for det in filtered_detections])
    class_names = [det['class_name'] for det in filtered_detections]
    class_ids = np.array(list(range(len(class_names))))
    
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]
    
    # Use original OpenCV image for visualization
    img = original_cv_image.copy()
    
    # Create detections object
    detections = sv.Detections(
        xyxy=xyxy_boxes,
        class_id=class_ids
    )
    
    # Annotate image
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    
    # Save annotated image
    output_filename = f"{IMG_NAME}_tiled_detections_bx_thr{BOX_THRESHOLD}_txt_thr{TEXT_THRESHOLD}_no_crops"
    if USE_SUPER_RESOLUTION:
        output_filename += f"_with_SR_{SR_CROP_SIZE}"
    output_filename += ".jpg"
    cv2.imwrite(os.path.join(OUTPUT_DIR, output_filename), annotated_frame)
    print(f"Saved annotated image to {os.path.join(OUTPUT_DIR, output_filename)}")

# Save results as JSON
if DUMP_JSON_RESULTS and filtered_detections:
    results = {
        "image_path": img_path,
        "original_dimensions": {"width": orig_w, "height": orig_h},
        "tiling_used": USE_TILING,
        "crop_size": CROP_SIZE if USE_TILING else None,
        "overlap": OVERLAP if USE_TILING else None,
        "total_crops": len(crops) if USE_TILING else 1,
        "super_resolution_used": USE_SUPER_RESOLUTION and sr_model is not None,
        "sr_scale_factor": SR_SCALE_FACTOR if USE_SUPER_RESOLUTION else None,
        "sr_model": SR_MODEL if USE_SUPER_RESOLUTION else None,
        "annotations": [
            {
                "class_name": det['class_name'],
                "bbox": det['bbox'],
                "confidence": det['confidence'],
                "score": det['score'],
                "used_super_resolution": det.get('used_super_resolution', False)
            }
            for det in filtered_detections
        ],
        "box_format": "xyxy",
    }
    
    # with open(os.path.join(OUTPUT_DIR, f"{IMG_NAME}_tiled_results.json"), "w") as f:
    #     json.dump(results, f, indent=4)
    
    print(f"Saved JSON results to {os.path.join(OUTPUT_DIR, f'{IMG_NAME}_tiled_results.json')}")

print(f"\nProcessing complete!")
print(f"- Found {len(filtered_detections)} objects total")
print(f"- YOLO labels saved to: {IMG_NAME}.txt")
if USE_SUPER_RESOLUTION and sr_model is not None:
    print(f"- Super resolution applied with {SR_SCALE_FACTOR}x scale using {SR_MODEL}")
    print(f"- Annotated image saved to: {IMG_NAME}_tiled_detections_with_SR.jpg")
else:
    print(f"- Annotated image saved to: {IMG_NAME}_tiled_detections.jpg")
if SAVE_CROPS:
    print(f"- {len(crops)} original crop images saved to: outputs/crops/")
if SAVE_SR_CROPS and USE_SUPER_RESOLUTION and sr_model is not None:
    print(f"- {len(crops)} super resolution crop images saved to: outputs/sr_crops/")
if DUMP_JSON_RESULTS:
    print(f"- JSON results saved to: {IMG_NAME}_tiled_results.json")
print(f'\nEnd time: {(time()-start):.2f} seconds')