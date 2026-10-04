import json
import random
from collections import defaultdict
from pathlib import Path

if __name__ == "__main__":
    data_dir = Path("data/DLRSD")
    images_dir = data_dir / "Images"
    labels_dir = data_dir / "Labels"
    random.seed(42)
    pairs_list = []
    train_pairs_list = []
    test_pairs_list = []
    classes = defaultdict(list)

    for image_class_dir in sorted(images_dir.iterdir()):
        if not image_class_dir.is_dir():
            continue
        else:
            for image_path in sorted(image_class_dir.glob("*.tif")):
                label_path = labels_dir / image_class_dir.name / (image_path.stem + ".png")
                if label_path.exists():
                    pairs_list.append((str(image_path), str(label_path)))

    assert len(pairs_list) == 2100, f"expected 2100 pairs, got {len(pairs_list)} pairs"

    for image_path, label_path in pairs_list:
        class_name = Path(image_path).parent.name
        classes[class_name].append((image_path, label_path))

    for class_name in sorted(classes):
        class_pairs = classes[class_name]
        random.shuffle(class_pairs)

        # 80% for training, 20% for testing 
        split_data = int(0.8 * len(class_pairs))     
        train_pairs_list.extend(class_pairs[:split_data])
        test_pairs_list.extend(class_pairs[split_data:])

    assert len(classes) == 21

    with open("train_split.json", "w") as f:
        json.dump(train_pairs_list, f, indent=2)

    with open("test_split.json", "w") as f:
        json.dump(test_pairs_list, f, indent=2)

    print(f"train data: {len(train_pairs_list)}, test data: {len(test_pairs_list)}")
