import pandas as pd

df = pd.read_csv("frames_labeled.csv")
print("frames per split:")
print(df["split"].value_counts(), "\n")

for col in ["boredom", "engagement", "confusion", "frustration"]:
    print(col, "levels (0 to 3):")
    print(df.groupby("split")[col].value_counts().unstack(fill_value=0), "\n")

print("people per split:")
print(df.groupby("split")["person"].nunique())