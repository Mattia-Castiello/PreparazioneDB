import os
import json
import cv2
import numpy as np
from collections import defaultdict

def create_masks(coco_json_path, output_dir):
    """
    Generate binary mask images from COCO-style polygon segmentations.

    Parameters:
    - coco_json_path: Path to the COCO annotations JSON file.
    - output_dir: Directory to save generated mask images.

    For each image entry, a mask is created where each annotation's polygon is filled
    with its category ID (or 255 for ignore regions), and saved as a PNG.
    """
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    # Map images by ID and group annotations per image
    images = {img['id']: img for img in coco.get('images', [])}
    img_to_anns = defaultdict(list)
    for ann in coco.get('annotations', []):
        img_to_anns[ann['image_id']].append(ann)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each image
    for img_id, anns in img_to_anns.items():
        if img_id not in images:
            print(f"Warning: Image ID {img_id} not found in image list, skipping.")
            continue

        img_info = images[img_id]
        height, width = img_info['height'], img_info['width']
        mask = np.zeros((height, width), dtype=np.uint8)

        for ann in anns:
            # Determine fill color: category_id+1, or 255 for crowd
            color = ann.get('category_id', 0) + 1
            if ann.get('iscrowd', 0):
                color = 255

            # Fill each polygon in segmentation
            for seg in ann.get('segmentation', []):
                try:
                    poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [poly], color)
                except Exception as e:
                    print(f"Failed to fill polygon for annotation {ann.get('id')}: {e}")

        # Save mask image
        base = os.path.splitext(os.path.basename(img_info['file_name']))[0]
        out_path = os.path.join(output_dir, f"{base}_mask.png")
        cv2.imwrite(out_path, mask)
        print(f"Saved mask for image {base} to {out_path}")


if __name__ == '__main__':
    # Example usage
    COCO_JSON = '/Users/mattiacastiello/Desktop/tesi/code/PreparazioneDB/Filejson/merged_annotations.json'
    OUTPUT_DIR = '/Users/mattiacastiello/Desktop/tesi/code/PreparazioneDB/Filejson/masks'
    create_masks(COCO_JSON, OUTPUT_DIR)
