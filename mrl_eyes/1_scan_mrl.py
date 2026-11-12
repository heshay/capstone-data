import os, re, hashlib
import pandas as pd
from PIL import Image

ROOT = "mrlEyes_2018_01"
pat = re.compile(r"^s(\d+)_(\d+)_(\d)_(\d)_(\d)_(\d)_(\d)_(\d+)\.png$")
rows, bad_names = [], []

for folder, _, files in os.walk(ROOT):
    for f in files:
        if not f.lower().endswith(".png"):
            continue
        m = pat.match(f)
        path = os.path.join(folder, f)
        if not m:
            bad_names.append(path); continue
        subj, num, gender, glasses, eye, refl, light, sensor = m.groups()
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                w, h = im.size; mode = im.mode
            with open(path, "rb") as fh:
                md5 = hashlib.md5(fh.read()).hexdigest()
            ok = True
        except Exception:
            w = h = None; mode = None; md5 = None; ok = False
        rows.append(dict(path=path, subject=f"s{subj}", image=int(num),
            gender=int(gender), glasses=int(glasses), eye_open=int(eye),
            reflections=int(refl), good_light=int(light), sensor=sensor,
            width=w, height=h, mode=mode, md5=md5, readable=ok,
            bytes=os.path.getsize(path)))

df = pd.DataFrame(rows).sort_values(["subject","image"]).reset_index(drop=True)
df.to_csv("mrl_all_raw.csv", index=False)
print("pngs parsed:", len(df), "| bad filenames:", len(bad_names))
print("unreadable:", (~df.readable).sum())
print("subjects:", df.subject.nunique())
print("sizes:\n", df.groupby(["width","height"]).size().sort_values(ascending=False).head(8))
print("modes:", df["mode"].value_counts().to_dict())
print("duplicate content (same md5):", df.md5.duplicated(keep=False).sum())
print("tiny files (<600B):", (df.bytes<600).sum())
for c in ["gender","glasses","eye_open","reflections","good_light","sensor"]:
    print(c, df[c].value_counts().to_dict())
print("eye_open by subject (min/max share):")
s = df.groupby("subject").eye_open.agg(["mean","size"])
print(s.describe().round(3))
