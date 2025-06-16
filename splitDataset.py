import json
import random

# Path to your merged COCO-style annotation file
input_json = '/Users/mattiacastiello/Desktop/tesi/code/PreparazioneDB/Filejson/merged_annotations.json'

# Load merged annotations
with open(input_json, 'r') as f:
    coco = json.load(f)

images = coco['images']
annotations = coco['annotations']
categories = coco.get('categories', [])

# Shuffle images for random split
random.seed(42)
random.shuffle(images)

# Split ratio (e.g., 80% train, 20% val)
train_ratio = 0.8
split_idx = int(len(images) * train_ratio)

train_images = images[:split_idx]
val_images = images[split_idx:]

# Build subsets
def build_subset(images_subset):
    img_ids = {img['id'] for img in images_subset}
    subset_annotations = [ann for ann in annotations if ann['image_id'] in img_ids]
    return {
        'images': images_subset,
        'annotations': subset_annotations,
        'categories': categories
    }

train_coco = build_subset(train_images)
val_coco = build_subset(val_images)

# Output file paths
train_json = '/Users/mattiacastiello/Desktop/tesi/code/PreparazioneDB/Filejson/train_annotations.json'
val_json = '/Users/mattiacastiello/Desktop/tesi/code/PreparazioneDB/Filejson/val_annotations.json'

# Save subsets
with open(train_json, 'w') as f:
    json.dump(train_coco, f, indent=2)
with open(val_json, 'w') as f:
    json.dump(val_coco, f, indent=2)

print(f'Successfully split into {len(train_images)} train and {len(val_images)} val images.')
print(f'Train annotations saved to {train_json}')
print(f'Val annotations saved to {val_json}')