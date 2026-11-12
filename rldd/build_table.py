import os, re
import pandas as pd

OUT = r"frames"
FPS = 1
CLASS_NAME = {0: "alert", 5: "low_vigilant", 10: "drowsy"}
PUBLISHABLE = {1,2,4,9,10,11,13,16,17,19,21,22,23,24,25,26,28,29,30,31,33,34,
               42,43,46,47,48,49,50,51,52,53,55,57,59,60}

rows = []
for fold in sorted(os.listdir(OUT)):
    for subj in sorted(os.listdir(os.path.join(OUT, fold))):
        for cls in sorted(os.listdir(os.path.join(OUT, fold, subj))):
            d = os.path.join(OUT, fold, subj, cls)
            label = int(cls[1:])
            m = re.search(r"\d+", fold)
            for k, f in enumerate(sorted(x for x in os.listdir(d) if x.endswith(".jpg"))):
                p = os.path.join(d, f)
                rows.append(dict(fold=fold, fold_num=int(m.group()) if m else None,
                                 subject=int(subj[1:]), label=label,
                                 class_name=CLASS_NAME[label],
                                 binary=(1 if label == 10 else 0 if label == 0 else None),
                                 second=k / FPS, publishable=int(subj[1:]) in PUBLISHABLE,
                                 path=p, bytes=os.path.getsize(p)))

df = pd.DataFrame(rows)
vid = df.groupby(["fold", "subject", "label"]).size()
print("frames:", len(df), "| videos:", len(vid), "| subjects:", df.subject.nunique())
print("subjects per fold:", df.groupby("fold").subject.nunique().to_dict())
per = df.groupby("subject").label.nunique()
print("subjects without all 3 classes:", per[per != 3].index.tolist())
print("frames per video:", vid.describe().round(0).to_dict())
print("videos with under 300 frames:", (vid < 300).sum())
print("tiny or blank frames (<1000 bytes):", (df.bytes < 1000).sum())
print("frames per class:", df.class_name.value_counts().to_dict())
print("publishable frames:", int(df.publishable.sum()), "of", len(df))

df.drop(columns="bytes").to_csv("rldd_frames.csv", index=False)
print("saved rldd_frames.csv")