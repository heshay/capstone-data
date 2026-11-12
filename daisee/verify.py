import pandas as pd, os

df = pd.read_csv("frames_labeled.csv")
missing = df[~df["path"].map(os.path.exists)]
small = df[df["path"].map(lambda p: os.path.exists(p) and os.path.getsize(p) < 2000)]
per_clip = df.groupby("clip").size()

print("rows:", len(df), "| clips:", df["clip"].nunique())
print("missing files:", len(missing))
print("suspiciously tiny (blank?) frames:", len(small))
print("clips with fewer than 8 frames:", (per_clip < 8).sum())
print("label values outside 0-3:", (~df[["boredom","engagement","confusion","frustration"]].isin([0,1,2,3]).all(axis=1)).sum())