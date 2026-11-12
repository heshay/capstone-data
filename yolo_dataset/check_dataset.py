import os
from collections import Counter
import pandas as pd

inv = pd.read_csv("inventory.csv")
inv["stem"] = inv.path.map(lambda p: os.path.splitext(os.path.basename(p))[0])
img = inv[inv.ext == ".jpg"]
lab = inv[inv.ext == ".txt"]

for s in ["train", "val", "test"]:
    a = set(img[img.split == s].stem)
    b = set(lab[lab.split == s].stem)
    print(s, "images without label:", len(a - b), "| labels without image:", len(b - a))

d = img[img.md5.duplicated(keep=False)]
print("\nduplicate images:", len(d), "in", d.md5.nunique(), "groups")
cross = img.groupby("md5").split.nunique()
print("identical images in more than one split:", int((cross > 1).sum()))

fields, classes, bad_range, empty = Counter(), Counter(), 0, 0
for _, r in lab.iterrows():
    with open(r.path) as f:
        lines = [l.split() for l in f.read().splitlines() if l.strip()]
    if not lines:
        empty += 1
        continue
    for parts in lines:
        fields[len(parts)] += 1
        classes[(r.split, parts[0])] += 1
        try:
            if len(parts) == 5 and not all(0 <= float(x) <= 1 for x in parts[1:]):
                bad_range += 1
        except ValueError:
            bad_range += 1

print("\nempty label files:", empty)
print("numbers per line (5 = normal box):", dict(fields))
print("boxes with values outside 0 to 1:", bad_range)
print(pd.Series(classes).unstack(0).fillna(0).astype(int).to_string())