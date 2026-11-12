import os, shutil, hashlib
from collections import defaultdict

SRC = r"raw"   
DST = r"cleaned_dataset"       
SPLITS = ["train", "val", "test"]  

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def clean_boxes(path):
    """Split lines holding several boxes, clip to 0-1, drop repeats and empty boxes."""
    boxes, fixed = set(), 0
    with open(path) as f:
        for line in f.read().splitlines():
            parts = line.split()
            if len(parts) > 5 or len(parts) % 5 != 0:
                fixed += 1
            for i in range(0, len(parts) - len(parts) % 5, 5):
                try:
                    c = int(float(parts[i]))
                    x, y, w, h = (float(v) for v in parts[i + 1:i + 5])
                except ValueError:
                    continue
                x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
                x2, y2 = min(1.0, x + w / 2), min(1.0, y + h / 2)
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue
                boxes.add((c, round((x1 + x2) / 2, 6), round((y1 + y2) / 2, 6),
                           round(x2 - x1, 6), round(y2 - y1, 6)))
    return sorted(boxes), fixed

entries, fixed_lines = [], 0
for s in SPLITS:
    img_dir = os.path.join(SRC, s, "images")
    for name in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() != ".jpg":
            continue
        boxes, fx = clean_boxes(os.path.join(SRC, s, "labels", stem + ".txt"))
        fixed_lines += fx
        entries.append((s, stem, os.path.join(img_dir, name), boxes))
print("images read:", len(entries))

groups = defaultdict(list)
for e in entries:
    groups[md5(e[2])].append(e)

keep, dup_removed, conflict_removed = [], 0, 0
for g in groups.values():
    if len(g) == 1:
        keep.append(g[0])
        continue
    if len({tuple(sorted(b[0] for b in e[3])) for e in g}) > 1: 
        conflict_removed += len(g)
        continue
    g.sort(key=lambda e: SPLITS.index(e[0]))
    keep.append(g[0])
    dup_removed += len(g) - 1

for s in SPLITS:
    os.makedirs(os.path.join(DST, s, "images"), exist_ok=True)
    os.makedirs(os.path.join(DST, s, "labels"), exist_ok=True)
for s, stem, img, boxes in keep:
    shutil.copy2(img, os.path.join(DST, s, "images", stem + ".jpg"))
    with open(os.path.join(DST, s, "labels", stem + ".txt"), "w") as f:
        for c, x, y, w, h in boxes:
            f.write(f"{c} {x} {y} {w} {h}\n")

print("duplicate copies removed:", dup_removed)
print("images removed for conflicting labels:", conflict_removed)
print("label lines repaired:", fixed_lines)
print("images kept:", len(keep))
for s in SPLITS:
    print(" ", s, sum(1 for k in keep if k[0] == s))