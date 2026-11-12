import os, hashlib
import pandas as pd
from PIL import Image

ROOT = "raw" 

rows = []
for folder, _, files in os.walk(ROOT):
    rel = os.path.relpath(folder, ROOT).split(os.sep)
    split = rel[0] if rel[0] != "." else ""
    label = rel[1] if len(rel) > 1 else ""
    for f in files:
        p = os.path.join(folder, f)
        ext = os.path.splitext(f)[1].lower()
        readable, w, h = None, None, None
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    w, h = im.size
                readable = True
            except Exception:
                readable = False
        with open(p, "rb") as fh:
            md5 = hashlib.md5(fh.read()).hexdigest()
        rows.append(dict(path=p, split=split, label=label, ext=ext,
                         bytes=os.path.getsize(p), readable=readable,
                         width=w, height=h, md5=md5))

df = pd.DataFrame(rows)
df.to_csv("inventory.csv", index=False)

print("total files:", len(df))
print("\nfile types:\n", df.ext.value_counts().to_string())
print("\nfiles per split:\n", df.split.value_counts().to_string())
print("\nfiles per split and class:\n", df.groupby(["split", "label"]).size().to_string())
print("\nunreadable images:", (df.readable == False).sum())
print("empty files:", (df.bytes == 0).sum())
print("image sizes (top 5):\n", df.groupby(["width", "height"]).size().sort_values(ascending=False).head(5).to_string())
print("\nduplicate files:", df.md5.duplicated(keep=False).sum())
cross = df.groupby("md5").split.nunique()
print("identical files appearing in more than one split:", (cross > 1).sum())