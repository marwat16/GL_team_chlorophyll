import json
import numpy as np
from PIL import Image

pairs = json.load(open("train_split.json"))
vals = set()
for _, lbl in pairs:
    vals |= set(np.unique(np.array(Image.open(lbl))).tolist())

print("Unique label values:", sorted(vals))
