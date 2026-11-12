import os, re, json, subprocess
import pandas as pd

ROOT = r"videos"   
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp")
PUBLISHABLE = {1,2,4,9,10,11,13,16,17,19,21,22,23,24,25,26,28,29,30,31,33,34,
               42,43,46,47,48,49,50,51,52,53,55,57,59,60}

def probe(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,r_frame_rate:format=duration",
           "-of", "json", path]
    try:
        j = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout)
        s = j["streams"][0]
        n, d = s["r_frame_rate"].split("/")
        return dict(width=s["width"], height=s["height"], fps=round(float(n) / float(d), 2),
                    seconds=round(float(j["format"]["duration"]), 1), playable=True)
    except Exception:
        return dict(width=None, height=None, fps=None, seconds=None, playable=False)

rows = []
for folder, _, files in os.walk(ROOT):
    for f in sorted(files):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in VIDEO_EXT:
            continue
        p = os.path.join(folder, f)
        parts = os.path.relpath(p, ROOT).split(os.sep)
        fold = next((x for x in parts if re.search("fold", x, re.I)), "")
        subject = next((x for x in parts[:-1] if x.isdigit()), "")
        label = stem if stem in ("0", "5", "10") else ""
        rows.append(dict(path=p, fold=fold, subject=subject, label=label,
                         ext=ext.lower(), gb=round(os.path.getsize(p) / 1e9, 2), **probe(p)))
        print("scanned", len(rows), flush=True)

if not rows:
    print("No videos found. Check ROOT.")
    raise SystemExit

df = pd.DataFrame(rows)
df["publishable"] = df.subject.map(lambda s: int(s) in PUBLISHABLE if s.isdigit() else None)

def flag(r):
    f = []
    if not r.playable: f.append("unplayable")
    elif r.seconds < 300 or r.seconds > 1200: f.append("odd length")
    if r.subject == "": f.append("no subject")
    if r.label == "": f.append("no label")
    return ", ".join(f)
df["flag"] = df.apply(flag, axis=1)
df.to_csv("rldd_inventory.csv", index=False)

print("\nvideos:", len(df), "| folds:", sorted(df.fold.unique()))
print("per class:", df.label.value_counts().to_dict())
per = df.groupby("subject").size()
print("subjects:", len(per), "| subjects without exactly 3 videos:", per[per != 3].index.tolist())
print("minutes per video:", (df.seconds / 60).describe().round(1).to_dict())
print("fps:", df.fps.value_counts().head(5).to_dict())
print("resolutions:", df.groupby(["width", "height"]).size().sort_values(ascending=False).head(5).to_dict())
print("flagged videos:", (df.flag != "").sum())
print(df[df.flag != ""][["path", "flag"]].head(10).to_string())
print("\nfirst paths:"); print(df.path.head(4).to_string())