import argparse
import json
import os
from pathlib import Path
import random
import runpy
import sys
import shutil
import numpy as np
import torch
from PIL import Image

if __name__ == "__main__":
    seed = 42
    num_of_classes = 17         
    ma_unet_classes = 18        
    this_folder = Path(__file__).resolve().parent
    ma_unet_folder = this_folder / "ma_unet_repo"
    voc_folder = ma_unet_folder / "VOCdevkit" / "VOC2007"
    logs_folder = ma_unet_folder / "logs"
    train_data_file = this_folder / "train_split.json"
    test_data_file = this_folder / "test_split.json"

    parser = argparse.ArgumentParser()
    # parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--subset", type=int, default=0)  
    args = parser.parse_args()
 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
 
    # load split data:
    with open(train_data_file) as f:
        train_data_pairs = [(this_folder / image_path, this_folder / label_path)
                            for image_path, label_path in json.load(f)]
    with open(test_data_file) as f:
        test_data_pairs = [(this_folder / image_path, this_folder / label_path)
                           for image_path, label_path in json.load(f)]
 
    # only for test runs
    if args.subset:
        rng = random.Random(0)
        train_data_pairs = rng.sample(train_data_pairs, args.subset)
        test_data_pairs = rng.sample(test_data_pairs, args.subset)
 
    images_folder = voc_folder / "JPEGImages"
    labels_folder = voc_folder / "SegmentationClass"
    lists_folder = voc_folder / "ImageSets" / "Segmentation"
    for folder in (images_folder, labels_folder, lists_folder):
        folder.mkdir(parents=True, exist_ok=True)
 
    for image_path, label_path in train_data_pairs + test_data_pairs:
        image_link = images_folder / (image_path.stem + ".jpg")
        label_link = labels_folder / (label_path.stem + ".png")
        for link, target in ((image_link, image_path), (label_link, label_path)):
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target.resolve())
 
    with open(lists_folder / "train.txt", "w") as f:
        f.write("\n".join(image_path.stem for image_path, _ in train_data_pairs) + "\n")
    with open(lists_folder / "val.txt", "w") as f:
        f.write("\n".join(image_path.stem for image_path, _ in test_data_pairs) + "\n")

    # start from an empty logs folder
    if logs_folder.exists():
        shutil.rmtree(logs_folder)
 
    og_folder = Path.cwd()
    sys.path.insert(0, str(ma_unet_folder))
    os.chdir(ma_unet_folder)
    try:
        runpy.run_path("train.py", run_name="__main__")
    finally:
        os.chdir(og_folder)

    checkpoints = sorted(logs_folder.glob("ep*.pth"))
    final_checkpoint = checkpoints[-1]

    from unet import Unet
    model = Unet(model_path=str(final_checkpoint), num_classes=ma_unet_classes,
                 cuda=torch.cuda.is_available())

    # evaluation:
    intersection = np.zeros(num_of_classes)
    union = np.zeros(num_of_classes)
 
    for image_path, label_path in test_data_pairs:
        image = Image.open(image_path)                     
        prediction = np.array(model.get_miou_png(image))     
 
        label = Image.open(label_path)
        label = np.array(label)
 
        for class_number in range(1, num_of_classes + 1):
            predicted_pixels = prediction == class_number
            real_pixels = label == class_number
 
            same_pixels = predicted_pixels & real_pixels
            all_pixels = predicted_pixels | real_pixels
 
            intersection[class_number - 1] += same_pixels.sum()
            union[class_number - 1] += all_pixels.sum()

    total_iou = 0
    counted_classes = 0
    for i in range(num_of_classes):
        if union[i] > 0:
            iou = intersection[i] / union[i]
            total_iou += iou
            counted_classes += 1
 
    miou = total_iou / counted_classes
    print(f"mIoU: {miou:.4f}")
