import os
import random
import json

DATA_DIR = "data/DLRSD"
IMAGES_DIR = os.path.join(DATA_DIR, "Images")
LABELS_DIR = os.path.join(DATA_DIR, "Labels")

def build_all_pairs():
    """Walk through Images/ and Labels/, pairing each image with its label mask."""
    all_pairs = []
    class_folders = sorted(os.listdir(IMAGES_DIR))
    for class_name in class_folders:
        image_class_dir = os.path.join(IMAGES_DIR, class_name)
        label_class_dir = os.path.join(LABELS_DIR, class_name)
        if not os.path.isdir(image_class_dir):
            continue
        for filename in sorted(os.listdir(image_class_dir)):
            image_path = os.path.join(image_class_dir, filename)
            base_name = os.path.splitext(filename)[0]
            label_filename = base_name + ".png"
            label_path = os.path.join(label_class_dir, label_filename)
            if os.path.exists(label_path):
                all_pairs.append((image_path, label_path))
            else:
                print(f"WARNING: no label found for {image_path}")
    return all_pairs

def main():
    random.seed(42)
    all_pairs = build_all_pairs()
    print(f"Total pairs found: {len(all_pairs)}")   # expect 2100

    # group by scene category (the folder name)
    by_class = defaultdict(list)
    for image_path, label_path in all_pairs:
        class_name = os.path.basename(os.path.dirname(image_path))
        by_class[class_name].append((image_path, label_path))

    # 80/20 split within each category
    train_pairs, test_pairs = [], []
    for class_name in sorted(by_class):
        pairs = by_class[class_name]
        random.shuffle(pairs)
        split_idx = int(0.8 * len(pairs))
        train_pairs += pairs[:split_idx]
        test_pairs += pairs[split_idx:]

    print(f"Categories: {len(by_class)}")                           # expect 21
    print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")    # expect 1680 / 420

    with open("train_split.json", "w") as f:
        json.dump(train_pairs, f, indent=2)
    with open("test_split.json", "w") as f:
        json.dump(test_pairs, f, indent=2)

if __name__ == "__main__":
    main()
