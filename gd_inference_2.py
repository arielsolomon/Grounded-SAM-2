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
import glob
from tqdm import tqdm

torch.cuda.empty_cache() 

"""
Hyper parameters
"""
TEXT_PROMPT = "car. building."
IMG_PATH = "/work/data/china_env_dataset/mixed_images/images/"
IMG_PATH_LIST = []
[IMG_PATH_LIST.append(file) for file in glob.glob(os.path.join(IMG_PATH, '*.bmp'))]
IMG_PATH_LIST = IMG_PATH_LIST[:15]

# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
def run_inf(TEXT_PROMPT,IMG_PATH_LIST):

    SAM2_CHECKPOINT = "/work/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    GROUNDING_DINO_CONFIG = "/work/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = os.path.dirname(os.path.abspath(__file__))+"/gdino_checkpoints/groundingdino_swint_ogc.pth"
    BOX_THRESHOLD = 0.1
    TEXT_THRESHOLD = 0.1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = Path("outputs")
    DUMP_JSON_RESULTS = True

    # create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    text = TEXT_PROMPT
    for file in tqdm(IMG_PATH_LIST):
        img_path = file

        image_source, image = load_image(img_path)
        
        sam2_predictor.set_image(image_source)

        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=text,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE
        )

        # process the box prompt for SAM 2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Add a check to handle the case of no detections from Grounding DINO
        if len(input_boxes) > 0:
            # ONLY wrap SAM2 prediction with the autocast block
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                masks, scores, logits = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )

            """
            Post-process the output of the model to get the masks, scores, and logits for visualization
            """
            # convert the shape to (n, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)

            confidences = confidences.numpy().tolist()
            class_names = labels

            class_ids = np.array(list(range(len(class_names))))

            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence
                in zip(class_names, confidences)
            ]

            """
            Visualize image with supervision useful API
            """
            img = cv2.imread(img_path)
            detections = sv.Detections(
                xyxy=input_boxes,  # (n, 4)
                mask=masks.astype(bool),  # (n, h, w)
                class_id=class_ids
            )

            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            img_name = os.path.basename(file)
            cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), annotated_frame)

            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            cv2.imwrite(os.path.join(OUTPUT_DIR, img_name.split('.')[0]+"mask.bmp"), annotated_frame)

            """
            Dump the results in standard format and save as json files
            """

            def single_mask_to_rle(mask):
                rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                rle["counts"] = rle["counts"].decode("utf-8")
                return rle

            if DUMP_JSON_RESULTS:
                # convert mask into rle format
                mask_rles = [single_mask_to_rle(mask) for mask in masks]

                input_boxes = input_boxes.tolist()
                scores = scores.tolist()
                # save the results in standard format
                results = {
                    "image_path": img_path,
                    "annotations" : [
                        {
                            "class_name": class_name,
                            "bbox": box,
                            "segmentation": mask_rle,
                            "score": score,
                        }
                        for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
                    ],
                    "box_format": "xyxy",
                    "img_width": w,
                    "img_height": h,
                }
                
                with open(os.path.join(OUTPUT_DIR, img_name.split('.')[0]+".json"), "w") as f:
                    json.dump(results, f, indent=4)
        else:
            print(f"No objects detected for image: {file}")
run_inf(TEXT_PROMPT,IMG_PATH_LIST)