import os, subprocess
import pandas as pd

INVENTORY = "rldd_inventory.csv"
OUT = r"frames"
FPS = 1          
SIZE = 224       
LIMIT = None      

df = pd.read_csv(INVENTORY, dtype=str).fillna("")
df = df[(df.playable == "True") & (df.subject != "") & (df.label != "")]
if LIMIT:
    df = df.head(LIMIT)

vf = (f"fps={FPS},scale={SIZE}:{SIZE}:force_original_aspect_ratio=decrease,"
      f"pad={SIZE}:{SIZE}:(ow-iw)/2:(oh-ih)/2")

for i, r in enumerate(df.itertuples(), 1):
    out_dir = os.path.join(OUT, r.fold or "nofold", f"s{int(r.subject):02d}", f"c{r.label}")
    marker = os.path.join(out_dir, ".done")
    if os.path.exists(marker):
        print(f"[{i}/{len(df)}] already done, skipping", flush=True)
        continue
    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):           
        os.remove(os.path.join(out_dir, old))
    print(f"[{i}/{len(df)}] fold {r.fold}, subject {r.subject}, class {r.label}", flush=True)
    res = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", r.path, "-vf", vf,
                          "-q:v", "3", os.path.join(out_dir, "f_%04d.jpg")])
    if res.returncode == 0:
        open(marker, "w").close()
    else:
        print("   FAILED:", r.path)
print("finished")