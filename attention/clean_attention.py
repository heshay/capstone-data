import os
import pandas as pd

ROOT = r"C:\path\to\Students Attention Detection Dataset"   # folder holding the 3 CSV files
OUT = "."                                                    # where the clean files are saved

def load(name):
    # keep_default_na=False: the word "None" in head_pose / gaze_direction is a real
    # category (nothing detected), not a missing value, so pandas must not turn it into NaN
    df = pd.read_csv(os.path.join(ROOT, name), keep_default_na=False, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(exclude="number").columns:
        df[c] = df[c].astype(str).str.strip().str.lower()
    return df

def add_split(df):
    # rows are consecutive video frames (neighbours are almost identical), so a random
    # split would leak. Cut into three consecutive blocks instead: 70% / 15% / 15%.
    n = len(df)
    df.insert(0, "row_id", range(n))
    df["split"] = "train"
    df.loc[df.row_id >= int(n * 0.70), "split"] = "val"
    df.loc[df.row_id >= int(n * 0.85), "split"] = "test"
    return df

# v1 is exactly the first 4,000 rows of v2, so it is dropped to avoid counting rows twice
v1, v2 = load("attention_detection_dataset_v1.csv"), load("attention_detection_dataset_v2.csv")
print("v1 contained in v2:", v1.equals(v2.iloc[:len(v1)].reset_index(drop=True)))

# v2 and v3 have different columns and coordinate scales, so they stay separate files
for name, src in [("v2", "attention_detection_dataset_v2.csv"), ("v3", "attention_detection_dataset_v3.csv")]:
    df = add_split(load(src))
    num = df.select_dtypes("number").columns
    assert df.isna().sum().sum() == 0 and df.label.isin([0, 1]).all()
    assert (df[[c for c in num if c.endswith(("_w", "_h"))]] >= 0).all().all()
    df.to_csv(os.path.join(OUT, f"attention_{name}_clean.csv"), index=False)
    print(f"\n{name}: {len(df)} rows -> attention_{name}_clean.csv")
    print(" label share by split:\n", df.groupby("split").label.mean().round(3).to_string())
    print(" rows per split:", df.split.value_counts().to_dict())
