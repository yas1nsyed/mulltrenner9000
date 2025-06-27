from ultralytics import YOLO
import cv2
import numpy as np
import os
from huggingface_hub import hf_hub_download

REPO_ID = "Yas1n/mulltrenner_yolov11"
FILENAME = "best_heavy_59classes.pt"

# Waste Bin Mapping (German Mülltrennung System)
bin_map = {
    "Yellow Bin (Gelbe Tonne)": [
        "Aluminium blister pack", "Aluminium foil", "Carded blister pack", "Clear plastic bottle",
        "Disposable food container", "Disposable plastic cup", "Drink Carton", "Drink can",
        "Foam cup", "Meal carton", "Metal lid", "Metal bottle cap", "Other plastic bottle",
        "Other plastic container", "Other plastic cup", "Other plastic wrapper", "Plastic bottle cap",
        "Plastic film", "Plastic glooves", "Plastic lid", "Plastic straw", "Polypropylene bag",
        "Pop tab", "Single-use carrier bag", "Six pack rings", "Spread tub", "Squeezable tube",
        "Tupperware"
    ],
    "Grey Bin (Restmüll)": [
        "Cigarette", "Garbage bag", "Shoe", "Unlabeled litter", "Plastified paper bag",
        "Styrofoam piece", "Rope & strings", "Foam food container", "Other plastic",
        "Pizza box", "Tissues"
    ],
    "Green Bin (Biotonne)": [
        "Food waste"
    ],
    "Blue Bin (Papiertonne)": [
        "Egg carton", "Normal paper", "Other Carton", "Paper Bag", "Paper cup", "Paper straw",
        "Pizza box", "Toilet tube", "Magazine paper", "Wrapping paper", "Corrugated carton"
    ],
    "Glascontainer": [
        "Glass bottle", "Glass cup", "Glass jar", "Broken glass"
    ],
    "Hazardous Waste (Sondermüll)": [
        "Battery", "Aerosol", "Scrap metal"
    ],
    "Deposit Return (Pfand)": [
        "Drink can", "Clear plastic bottle", "Glass bottle", "Food Can"
    ]
}

# Bin colors in RGB format (for OpenCV)
bin_colors = {
    "Yellow Bin (Gelbe Tonne)": (255, 255, 0),       # Yellow
    "Grey Bin (Restmüll)": (128, 128, 128),          # Gray
    "Green Bin (Biotonne)": (0, 255, 0),             # Green
    "Blue Bin (Papiertonne)": (0, 0, 250),           # Blue (bright)
    "Glascontainer": (0, 0, 0),                      # Black
    "Hazardous Waste (Sondermüll)": (255, 0, 0),     # Red
    "Deposit Return (Pfand)": (255, 0, 255)          # Purple/Magenta
}

class_to_bin = {cls: bin_type for bin_type, cls_list in bin_map.items() for cls in cls_list} #class name gives bin type

def display_result(image_to_segment):

    # Load model
    model = YOLO(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    )

    # Resize image
    image = cv2.resize(image_to_segment, (640,640))

    # Run prediction
    results = model(image_to_segment)
    result = results[0]

    # Get class names
    class_names = model.names

    # Inverse mapping: class name to bin
    class_to_bin = {cls: bin_name for bin_name, class_list in bin_map.items() for cls in class_list}

    # Handle case with no trash
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        overlay = image.copy()
        cv2.putText(overlay, "No trash detected!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA)
        return (overlay) 

    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()     # [N, 4]
    scores = result.boxes.conf.cpu().numpy()    # [N]
    class_ids = result.boxes.cls.cpu().numpy()  # [N]

    overlay = image.copy()

    for i in range(len(class_ids)):
        class_id = int(class_ids[i])
        score = scores[i]
        mask = masks[i]
        box = boxes[i].astype(int)

        class_name = class_names[class_id]

        bin_type = next((bin_name for bin_name, items in bin_map.items() if class_name in items), "abcd")
        color = bin_colors.get(bin_type, (255, 255, 255))  # fallback to white

        # Apply colored mask (blended)
        mask_3c = np.stack([mask] * 3, axis=-1)  # [H, W, 3]
        color_array = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
        colored_mask = (mask_3c * color_array).astype(np.uint8)

        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

        # Draw bounding box
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Draw label (class name + score + bin)
        label = f"{class_name} ({int(score * 100)}%)\n{bin_type}"
        for j, line in enumerate(label.split("\n")):
            text_pos = (box[0], box[1] - 10 - 20 * j)
            cv2.putText(overlay, line, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    
    return (overlay)
