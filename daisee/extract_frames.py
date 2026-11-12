import os, subprocess
import pandas as pd

ROOT = "DataSet"
LABELS_CSV = "AllLabels.csv"
SPLIT_DIR = "DataSet"
OUT = "frames"
FPS = 1
LIMIT = None                        

def stem(name):
    return os.path.splitext(os.path.basename(name.strip()))[0]


videos = {}
for folder, _, files in os.walk(ROOT):
    for f in files:
        if f.lower().endswith((".avi", ".mp4")):
            videos[stem(f)] = os.path.join(folder, f)
print("videos found:", len(videos))

labels = pd.read_csv(LABELS_CSV)
labels.columns = [c.strip() for c in labels.columns]
labels["clip"] = labels["ClipID"].astype(str).map(stem)

split_of = {}
for name, split in [("train.txt", "Train"), ("validation.txt", "Validation"), ("test.txt", "Test")]:
    path = os.path.join(SPLIT_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    split_of[stem(line)] = split
labels["split"] = labels["clip"].map(split_of)

print("labels without a video:", (~labels["clip"].isin(videos)).sum())
print("labels without a split:", labels["split"].isna().sum())
labels = labels[labels["clip"].isin(videos) & labels["split"].notna()]
if LIMIT:
    labels = labels.head(LIMIT)

rows = []
for _, r in labels.iterrows():
    clip = r["clip"]
    out_dir = os.path.join(OUT, r["split"], clip)
    if not os.path.isdir(out_dir) or not os.listdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", videos[clip],
                        "-vf", f"fps={FPS},scale=224:224",
                        os.path.join(out_dir, "f_%02d.jpg")])
    for f in sorted(os.listdir(out_dir)):
        rows.append({"split": r["split"], "person": clip[:6], "clip": clip,
                     "path": os.path.join(out_dir, f),
                     "boredom": r["Boredom"], "engagement": r["Engagement"],
                     "confusion": r["Confusion"], "frustration": r["Frustration"]})

pd.DataFrame(rows).to_csv("frames_labeled.csv", index=False)
print("done:", len(rows), "frames")