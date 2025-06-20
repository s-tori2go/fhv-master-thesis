import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
import requests
import torch
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import supervision as sv
import json
import torch
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from groundingdino.util.inference import load_model, Model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 0.1 Set image directories
DATA_DIR = "../../data/outfits/positive"
PERSON_DIR = "../../data/segmented_person"
ITEMS_DIR = "../../data/segmented_items"
image_folders = [DATA_DIR, PERSON_DIR, ITEMS_DIR]

# 0.2 Set model directories
SEGMENT_ANYTHING_MODEL_TYPE = "vit_h"
SEGMENT_ANYTHING_CHECKPOINT_PATH = "../../models/sam_vit_h.pth"
SEGMENT_ANYTHING_DOWNLOAD_URL = "https://hf-mirror.com/HCMUE-Research/SAM-vit-h/resolve/main/sam_vit_h_4b8939.pth?download=true"
GROUNDING_DINO_CONFIG_PATH = "../../groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "../../groundingdino/weights/groundingdino_swint_ogc.pth"
GROUNDING_DINO_CONFIG_URL = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

def download_file(url, path):
    """Download file with progress tracking"""
    print(f"Downloading {os.path.basename(path)}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    print(f"\rProgress: {downloaded/total_size*100:.1f}%", end='')
        print(f"\n✅ Download complete: {path}")

def load_image(image_selection):
    # Step 1: Load and display image

    # Create image folders
    for folder in image_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 1.3 Load image
    image_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not image_files:
        raise ValueError("No images found in the directory!")
    image_path = os.path.join(DATA_DIR, image_files[image_selection])
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # 1.4 Convert for visualization
    # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 1.5 Display original image
    cv2.startWindowThread()
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

def load_image_without_bg(image):
    # Step 2: Remove background
    image_without_bg = remove(image)

    # 2.1 Create white background
    height, width = image_without_bg.shape[:2]
    person_with_white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 2.2 Blend using alpha channel
    alpha_channel = image_without_bg[:, :, 3] / 255.0
    for c in range(3):  # R, G, B channels
        person_with_white_bg[:, :, c] = (
                image_without_bg[:, :, c] * alpha_channel +
                person_with_white_bg[:, :, c] * (1 - alpha_channel)
        ).astype(np.uint8)

    # 2.3 Save and display
    cv2.startWindowThread()
    cv2.imshow("Segmented Person", person_with_white_bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return person_with_white_bg

def load_sam_model():
    # 0.2 Download SAM checkpoint if missing
    if not os.path.exists(SEGMENT_ANYTHING_CHECKPOINT_PATH):
        download_file(SEGMENT_ANYTHING_DOWNLOAD_URL, SEGMENT_ANYTHING_CHECKPOINT_PATH)
    else:
        print(f"✅ SAM checkpoint exists: {SEGMENT_ANYTHING_CHECKPOINT_PATH}")

    # 0.4 Load SAM model
    if not torch.cuda.is_available():
        print("Warning: Using CPU - performance will be degraded")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        sam = sam_model_registry[SEGMENT_ANYTHING_MODEL_TYPE](
            checkpoint=SEGMENT_ANYTHING_CHECKPOINT_PATH
        ).to(device)
        sam_predictor = SamPredictor(sam)
        print("✅ SAM loaded successfully")
    except Exception as e:
        print(f"❌ SAM loading failed: {str(e)}")
        # Handle missing dependencies
        if "No module named 'segment_anything'" in str(e):
            print("Installing segment-anything...")
            os.system("pip install git+https://github.com/facebookresearch/segment-anything.git")
            # Retry after install
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry[SEGMENT_ANYTHING_MODEL_TYPE](
                checkpoint=SEGMENT_ANYTHING_CHECKPOINT_PATH
            ).to(device)
            sam_predictor = SamPredictor(sam)
    return sam_predictor

def load_dino_model():
    # 0.3 Download Grounding DINO files if missing
    if not os.path.exists(GROUNDING_DINO_CONFIG_PATH):
        download_file(GROUNDING_DINO_CONFIG_URL, GROUNDING_DINO_CONFIG_PATH)
    else:
        print(f"✅ Grounding DINO config exists: {GROUNDING_DINO_CONFIG_PATH}")

    if not os.path.exists(GROUNDING_DINO_CHECKPOINT_PATH):
        download_file(GROUNDING_DINO_CHECKPOINT_URL, GROUNDING_DINO_CHECKPOINT_PATH)
    else:
        print(f"✅ Grounding DINO checkpoint exists: {GROUNDING_DINO_CHECKPOINT_PATH}")

    # 0.5 Load Grounding DINO
    try:
        grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=device
        )
        print("✅ Grounding DINO loaded successfully")
    except Exception as e:
        print(f"❌ Grounding DINO loading failed: {str(e)}")
        if "No module named 'groundingdino'" in str(e):
            print("Installing GroundingDINO...")
            os.system("pip install git+https://github.com/IDEA-Research/GroundingDINO.git")
            # Retry after install
            from groundingdino.util.inference import load_model
            grounding_dino_model = Model(
                model_config_path=GROUNDING_DINO_CONFIG_PATH,
                model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
                device=device
            )
    return grounding_dino_model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--bert_base_uncased_path", type=str, required=False, help="bert_base_uncased model path, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    bert_base_uncased_path = args.bert_base_uncased_path

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not image_files:
        raise ValueError("No images found in the directory!")
    image_selection = 100
    image = load_image(image_selection)
    image_without_bg = load_image_without_bg(image)

    # load model
    sam_predictor = load_sam_model()
    grounding_dino_model = load_dino_model()

    # visualize raw image
    image_path = os.path.join(PERSON_DIR, image_files[image_selection])
    cv2.imwrite(f"{image_path}_segmented.jpg", image_without_bg)

    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, "grounded_sam_output.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    save_mask_data(output_dir, masks, boxes_filt, pred_phrases)