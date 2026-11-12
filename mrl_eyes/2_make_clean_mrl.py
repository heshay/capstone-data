import pandas as pd, numpy as np

df = pd.read_csv("mrl_all_raw.csv")
print("size range w:", df.width.min(), df.width.max(), "h:", df.height.min(), df.height.max())

# 1. drop exact duplicate images (keep first)
n0 = len(df)
dups = df[df.md5.duplicated(keep=False)]
print("duplicate groups that disagree on eye_open:",
      (dups.groupby("md5").eye_open.nunique() > 1).sum())
# identical images with CONFLICTING labels are label errors: drop every copy
conflict = dups.groupby("md5").eye_open.nunique()
conflict = conflict[conflict > 1].index
df = df[~df.md5.isin(conflict)]
print("dropped conflicting-label images:", n0 - len(df))
df = df.drop_duplicates("md5", keep="first")
print("removed duplicates:", n0 - len(df))

# 2. split BY PERSON (greedy, keeps eye_open balance reasonable)
rng = np.random.RandomState(42)
subs = df.groupby("subject").agg(n=("path","size"), open=("eye_open","mean")).reset_index()
subs = subs.sample(frac=1, random_state=42).sort_values("n", ascending=False)
target = {"train":0.70, "val":0.15, "test":0.15}
tot = len(df); filled = {k:0 for k in target}; assign = {}
for _, r in subs.iterrows():
    k = max(target, key=lambda s: target[s]*tot - filled[s])
    assign[r.subject] = k; filled[k] += r.n
df["split"] = df.subject.map(assign)

print("\nsplit sizes:\n", df.split.value_counts())
print("subjects per split:\n", df.groupby("split").subject.nunique())
print("eye_open share per split:\n", df.groupby("split").eye_open.mean().round(3))
print("glasses share per split:\n", df.groupby("split").glasses.mean().round(3))

out = df[["path","subject","split","eye_open","gender","glasses","reflections","good_light","sensor","width","height"]]
out.to_csv("mrl_clean.csv", index=False)
print("\nsaved mrl_clean.csv rows:", len(out))
