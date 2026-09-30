import json
from collections import Counter

import numpy as np
from PIL import Image

for split in ("train_split.json", "test_split.json"):
    with open(split) as f:
        pairs = json.load(f)

    values = set()
    sizes = Counter()
    mismatched = []
    for img_path, lbl_path in pairs:
        img = Image.open(img_path)
        lbl = Image.open(lbl_path)
        values |= set(np.unique(np.array(lbl)).tolist())
        sizes[img.size] += 1
        if img.size != lbl.size:
            mismatched.append(img_path)

    print(f"=== {split} ({len(pairs)} pairs) ===")
    print("Unique label values:", sorted(values))
    print("Image sizes (w, h):", dict(sizes))
    print("Image/label size mismatches:", len(mismatched))
    for p in mismatched[:5]:
        print("   ", p)
    print()
    