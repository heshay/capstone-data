"""Builds Maslak_EDA.ipynb from the cell sources below.

Run:  python build_notebook.py
Each cell is ("md" | "code", source). Keep the notebook logic here so it can be
reviewed as plain Python and regenerated without hand-editing JSON.
"""
import json
from pathlib import Path

CELLS = []


def md(src):
    CELLS.append(("markdown", src.strip("\n")))


def code(src):
    CELLS.append(("code", src.strip("\n")))


# ---------------------------------------------------------------------------
md(r"""
# Maslak (مسلك) — Smart Study Room · Exploratory Data Analysis

This notebook explores the five datasets that Maslak uses to recognise distraction signals during a study session. It starts from the **cleaned outputs** of the data-cleaning stage (repo [`heshay/capstone-data`](https://github.com/heshay/capstone-data)) and does not change them.

| Maslak signal | Dataset | Unit analysed here |
|---|---|---|
| Attention (head pose, gaze, phone, hands) | Students Attention Detection **V3** | 1 row = extracted webcam features |
| Phone use | **FPI-Det** (Face–Phone Interaction) | 1 image + YOLO boxes |
| Engagement / boredom / confusion / frustration | **DAiSEE** | 1 frame (10 per clip) |
| Drowsiness | **UTA-RLDD** | 1 frame (1 fps) |
| Eye closure (drowsiness helper) | **MRL Eye** | 1 eye crop |

**What every section checks:**
1. Structure and integrity: rows, subjects, duplicates, missing values.
2. Target distribution: overall and per split.
3. Split hygiene: whether a person or a video appears in more than one split.
4. Data that is unbalanced or redundant.
5. Image properties (when the images are available).
6. **Findings → modelling decisions**, collected automatically in a summary at the end.

> The indicators in these datasets are *behavioural signals*. They do not measure understanding or real concentration. Final accuracy has to be measured on recordings from the study-room setting itself.

**How to run:** `Runtime → Run all`. The tabular analysis always runs, because the cleaned CSVs come straight from GitHub. The image checks run only for the archives the notebook can find (see **0.2**). Any archive that is missing is skipped with a message and does not raise an error.
""")

md(r"""
## 0. Setup
### 0.1 Libraries and environment
""")

code(r"""
import os, sys, re, shutil, zipfile, subprocess, warnings, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from IPython.display import display, Markdown

warnings.filterwarnings("ignore", category=FutureWarning)
IN_COLAB = "google.colab" in sys.modules

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold", "axes.titlesize": 11})

print(f"Python {sys.version.split()[0]} | pandas {pd.__version__} | seaborn {sns.__version__} | "
      f"{'Google Colab' if IN_COLAB else 'local run'}")
""")

md(r"""
### 0.2 Configuration
Edit this cell only if your files are somewhere else.

* The **cleaned CSVs** are cloned from GitHub automatically.
* The **image archives** (`daisee_frames.zip`, `rldd_frames.zip`, `fpi_det_cleaned_dataset.zip`) are searched for in `ARCHIVE_SEARCH_DIRS`. On Colab, the simplest option is to put them in **Google Drive → `MyDrive/capstone-data-frames/`**. Another option is to paste a direct download link into `ARCHIVE_URLS`, or the share link of a OneDrive folder that holds all the zips into `ARCHIVE_FOLDER_URL`.
* **MRL Eye** is downloaded from its official page (≈340 MB) if it isn't found.
""")

code(r"""
SEED = 42
rng = np.random.default_rng(SEED)

WORK_DIR = Path(os.environ.get("EDA_WORK_DIR", "/content" if IN_COLAB else ".")).resolve()

# Cleaned tabular data (output of the data-cleaning stage)
REPO_URL    = "https://github.com/heshay/capstone-data.git"
REPO_COMMIT = ""          # optional: pin a commit hash, e.g. "623bd48", for reproducibility
REPO_DIR    = Path(os.environ.get("EDA_REPO_DIR", WORK_DIR / "capstone-data"))

# Image archives
RUN_IMAGE_EDA = True
MOUNT_DRIVE   = True      # Colab only: mounts Google Drive to look for the archives
ARCHIVES = {
    "daisee": "daisee_frames.zip",
    "rldd":   "rldd_frames.zip",
    "fpi":    "fpi_det_cleaned_dataset.zip",
    "mrl":    "mrlEyes_2018_01.zip",
}
ARCHIVE_URLS = {          # optional direct links (Google Drive share links work too)
    "daisee": "",
    "rldd":   "",
    "fpi":    "",
    "mrl":    "https://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip",
}
# optional: share link of a OneDrive folder that holds the zips above (files are matched by name).
# Ask the data-cleaning owner for the link; it is not stored in the repo because the images include faces.
ARCHIVE_FOLDER_URL = ""
ARCHIVE_SEARCH_DIRS = [
    "/content/drive/MyDrive/capstone-data-frames",
    "/content/drive/MyDrive",
    "/content",
    str(WORK_DIR),
]
if os.environ.get("EDA_OFFLINE"):          # testing: never download
    ARCHIVE_URLS = {k: "" for k in ARCHIVE_URLS}
    ARCHIVE_FOLDER_URL = ""
if os.environ.get("EDA_ARCHIVE_DIRS"):
    ARCHIVE_SEARCH_DIRS = os.environ["EDA_ARCHIVE_DIRS"].split(os.pathsep) + ARCHIVE_SEARCH_DIRS

IMG_PER_GROUP = 120       # images sampled per class for image statistics
DUP_SAMPLE_PER_SPLIT = 1500   # images per split for the near-duplicate audit (section 6)
NEAR_DUP_BITS = 4             # dHash distance (out of 64 bits) that makes a pair a duplicate *candidate*
DUP_MIN_CORR  = 0.97          # pixel correlation (32×32 grayscale) needed to confirm a candidate as a real duplicate
DATA_DIR = WORK_DIR / "eda_data"
OUT_DIR  = WORK_DIR / "eda_outputs"
FIG_DIR  = OUT_DIR / "figures"
for d in (DATA_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
print("work dir:", WORK_DIR)
""")

md(r"""
### 0.3 Helper functions
""")

code(r"""
FINDINGS = {}

def add_finding(key, dataset, finding, decision, evidence=""):
    # Register a finding and the modelling decision it implies (re-running a cell overwrites, never duplicates).
    FINDINGS[key] = dict(dataset=dataset, finding=finding, decision=decision, evidence=evidence)
    print(f"• [{dataset}] {finding}\n    → decision: {decision}")

def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=130)
    plt.show()
    plt.close(fig)

def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"

def header(text):
    display(Markdown(f"**{text}**"))

def check_columns(df, required, name):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name}: columns missing from the cleaned file: {missing}. "
                         "The cleaning output changed — update this notebook's column lists.")

def split_overlap(df, split_col, id_col):
    # How many ids appear in more than one split (0 = no leakage between splits).
    n_splits = df.groupby(id_col)[split_col].nunique()
    return int((n_splits > 1).sum())

def imbalance_ratio(counts):
    counts = pd.Series(counts)
    return float(counts.max() / max(counts.min(), 1))
""")

md(r"""
### 0.4 Load the cleaned data
""")

code(r"""
def run(cmd):
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{res.stderr.strip()}")
    return res.stdout.strip()

if not (REPO_DIR / "attention").exists():
    print("cloning", REPO_URL, "...")
    run(["git", "clone", "--quiet", REPO_URL, str(REPO_DIR)])
if REPO_COMMIT:
    run(["git", "-C", str(REPO_DIR), "checkout", "--quiet", REPO_COMMIT])
try:
    print("cleaned-data commit:", run(["git", "-C", str(REPO_DIR), "log", "-1", "--format=%h  %ad  %s", "--date=short"]))
except Exception:
    print("cleaned-data folder:", REPO_DIR, "(not a git checkout)")

FILES = {
    "attention_v3": "attention/attention_v3_clean.csv",
    "daisee":       "daisee/frames_labeled.csv",
    "rldd":         "rldd/rldd_frames.csv",
    "mrl":          "mrl_eyes/mrl_clean.csv",
}
data = {k: pd.read_csv(REPO_DIR / f) for k, f in FILES.items()}

check_columns(data["attention_v3"], ["label", "split", "head_pose", "gaze_on_screen", "phone_present"], "attention_v3")
check_columns(data["daisee"], ["split", "person", "clip", "path", "boredom", "engagement", "confusion", "frustration"], "daisee")
check_columns(data["rldd"], ["fold", "fold_num", "subject", "label", "class_name", "second", "publishable", "path"], "rldd")
check_columns(data["mrl"], ["path", "subject", "split", "eye_open", "glasses", "reflections", "good_light", "sensor", "width", "height"], "mrl")

inventory = pd.DataFrame([
    dict(dataset=k, file=FILES[k], rows=len(v), columns=v.shape[1],
         missing_cells=int(v.isna().sum().sum()), memory_mb=round(v.memory_usage(deep=True).sum() / 1e6, 1))
    for k, v in data.items()
])
display(inventory)
print("FPI-Det has no CSV in the repo — it is analysed from its YOLO label files in section 5.")
""")

md(r"""
### 0.5 Image archive helpers
These run only when an archive is found. Paths in the cleaned CSVs are relative and were written on Windows (`frames\Train\...`), so they are normalised and matched against the extracted folder automatically.
""")

code(r"""
def mount_drive():
    if IN_COLAB and MOUNT_DRIVE and not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception as e:
            print("Google Drive not mounted:", e)

def find_archive(key):
    for d in ARCHIVE_SEARCH_DIRS:
        p = Path(d) / ARCHIVES[key]
        if p.is_file():
            return p
    return None

def download(url, dest):
    print(f"downloading {dest.name} ...")
    tmp = dest.with_suffix(".part")
    if "drive.google.com" in url:
        import gdown
        gdown.download(url, str(tmp), quiet=True, fuzzy=True)
    else:
        if ("1drv.ms" in url or "onedrive.live" in url) and "download" not in url:
            url += ("&" if "?" in url else "?") + "download=1"
        for attempt in range(1, 4):           # large files: retry on network time-outs
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"  attempt {attempt} failed ({e}); retrying ...")
    if not zipfile.is_zipfile(tmp):
        tmp.unlink(missing_ok=True)
        raise ValueError("the link did not return a zip file (is it a direct download link?)")
    tmp.rename(dest)

_FOLDER_FILES = None

def onedrive_folder_files(share_url):
    # {file name: temporary direct download URL} for a public OneDrive folder share link.
    global _FOLDER_FILES
    if _FOLDER_FILES is None:
        import base64, json
        enc = "u!" + base64.urlsafe_b64encode(share_url.encode()).decode().rstrip("=")
        tok_req = urllib.request.Request("https://api-badgerp.svc.ms/v1.0/token", method="POST",
                                         data=json.dumps({"appId": "5cbed6ac-a083-4e14-b191-b4ba07653de2"}).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(tok_req, timeout=60) as r:
            token = json.load(r)["token"]
        req = urllib.request.Request(f"https://my.microsoftpersonalcontent.com/_api/v2.0/shares/{enc}/driveitem/children",
                                     headers={"Authorization": f"Badger {token}", "Prefer": "autoredeem"})
        with urllib.request.urlopen(req, timeout=60) as r:
            items = json.load(r).get("value", [])
        _FOLDER_FILES = {it["name"]: it["@content.downloadUrl"] for it in items if it.get("@content.downloadUrl")}
        print("OneDrive folder files:", sorted(_FOLDER_FILES))
    return _FOLDER_FILES

def prepare_images(key):
    # Return the extracted folder for an archive, or None (and say why) if it is not available.
    if not RUN_IMAGE_EDA:
        print(f"[{key}] image checks switched off (RUN_IMAGE_EDA = False)")
        return None
    out = DATA_DIR / key
    if (out / ".extracted").exists():
        return out
    src = find_archive(key)
    url = ARCHIVE_URLS.get(key, "")
    if src is None and not url and ARCHIVE_FOLDER_URL:
        try:
            url = onedrive_folder_files(ARCHIVE_FOLDER_URL).get(ARCHIVES[key], "")
        except Exception as e:
            print(f"[{key}] could not list the OneDrive folder: {e}")
    if src is None and url:
        src = DATA_DIR / ARCHIVES[key]
        try:
            download(url, src)
        except Exception as e:
            print(f"[{key}] download failed: {e}")
            src = None
    if src is None:
        mount_drive()
        src = find_archive(key)
    if src is None:
        print(f"[{key}] '{ARCHIVES[key]}' not found in {ARCHIVE_SEARCH_DIRS} → image checks skipped for this dataset.")
        return None
    print(f"[{key}] extracting {src.name} ({src.stat().st_size / 1e6:,.0f} MB) ...")
    out.mkdir(parents=True, exist_ok=True)
    try:
        if shutil.which("unzip"):
            subprocess.run(["unzip", "-q", "-o", str(src), "-d", str(out)], check=True)
        else:
            with zipfile.ZipFile(src) as z:
                z.extractall(out)
    except Exception as e:
        print(f"[{key}] could not extract: {e}")
        return None
    (out / ".extracted").touch()
    return out

def norm_rel(p):
    return Path(str(p).replace("\\", "/").lstrip("/"))

def resolve_paths(rel_paths, root, probe=200):
    # Find the base folder (and how many leading path parts to drop) that makes the CSV paths exist.
    rels = [norm_rel(p) for p in pd.Series(rel_paths).dropna().sample(min(probe, len(rel_paths)), random_state=SEED)]
    bases = [root]
    for lvl1 in sorted(p for p in root.iterdir() if p.is_dir()):
        bases.append(lvl1)
        bases += sorted(p for p in lvl1.iterdir() if p.is_dir())[:20]
    best = (0, None, 0)
    for base in bases:
        for strip in range(3):
            hits = sum(base.joinpath(*r.parts[strip:]).exists() for r in rels if len(r.parts) > strip)
            if hits > best[0]:
                best = (hits, base, strip)
    hits, base, strip = best
    if base is None:
        print("   could not match any CSV path inside", root)
        return None
    print(f"   paths resolved under '{base}' (dropping {strip} leading part(s)); probe hit-rate {hits}/{len(rels)}")
    return lambda p: base.joinpath(*norm_rel(p).parts[strip:])

def sample_existing(df, group_cols, n, to_abs, tries_factor=15):
    # Sample up to n rows per group whose image file really exists.
    rows = []
    for _, sub in df.groupby(group_cols, observed=True):
        sub = sub.sample(frac=1, random_state=SEED).head(n * tries_factor)
        got = 0
        for rec in sub.to_dict("records"):
            p = to_abs(rec["path"])
            if p.exists():
                rec["abs_path"] = str(p)
                rows.append(rec)
                got += 1
                if got >= n:
                    break
    return pd.DataFrame(rows)

def image_stats(paths, max_side=224):
    # Readability, size, brightness, contrast and sharpness (variance of the Laplacian) per image.
    from PIL import Image
    from scipy import ndimage
    out = []
    for p in paths:
        try:
            with Image.open(p) as im:
                w, h, mode = im.width, im.height, im.mode
                g = im.convert("L")
                g.thumbnail((max_side, max_side))
                a = np.asarray(g, dtype=np.float32)
            out.append(dict(readable=True, width=w, height=h, mode=mode, brightness=a.mean(),
                            contrast=a.std(), sharpness=ndimage.laplace(a).var()))
        except Exception:
            out.append(dict(readable=False, width=np.nan, height=np.nan, mode=None,
                            brightness=np.nan, contrast=np.nan, sharpness=np.nan))
    return pd.DataFrame(out)

def frame_diff(path_a, path_b, size=96):
    # Mean absolute pixel difference (0-255) between two frames — low values = near-duplicates.
    from PIL import Image
    imgs = []
    for p in (path_a, path_b):
        with Image.open(p) as im:
            imgs.append(np.asarray(im.convert("L").resize((size, size)), dtype=np.float32))
    return float(np.abs(imgs[0] - imgs[1]).mean())

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
_FACE_DETECTOR = None

def _get_face_detector():
    # YuNet (OpenCV's CNN face detector) when possible; Haar cascade as fallback. Returns (name, detect_fn) or None.
    global _FACE_DETECTOR
    if _FACE_DETECTOR is not None:
        return _FACE_DETECTOR
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed → face-visibility check skipped")
        return None
    if hasattr(cv2, "FaceDetectorYN"):
        try:
            model = DATA_DIR / "face_detection_yunet_2023mar.onnx"
            if not model.exists():
                urllib.request.urlretrieve(YUNET_URL, model)
            det = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.6)
            def detect(img):
                det.setInputSize((img.shape[1], img.shape[0]))
                _, faces = det.detect(img)
                return 0 if faces is None else len(faces)
            _FACE_DETECTOR = ("YuNet", detect)
            return _FACE_DETECTOR
        except Exception as e:
            print("YuNet unavailable:", e)
    if hasattr(cv2, "CascadeClassifier"):
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        def detect(img):
            g = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            return len(casc.detectMultiScale(g, 1.1, 5, minSize=(max(24, min(g.shape) // 8),) * 2))
        _FACE_DETECTOR = ("Haar cascade", detect)
        return _FACE_DETECTOR
    print("No OpenCV face detector available → face-visibility check skipped")
    return None

def face_counts(paths):
    # Number of faces found per image (NaN = unreadable); None if no detector is available.
    got = _get_face_detector()
    if got is None:
        return None
    import cv2
    name, detect = got
    out = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        out.append(np.nan if img is None else detect(img))
    print(f"face detector: {name} on {len(out)} images")
    return np.array(out, dtype=float)

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def dhash(path, size=8):
    # 64-bit difference hash: robust to resizing / re-encoding, so it finds visually near-identical images.
    from PIL import Image
    try:
        with Image.open(path) as im:
            a = np.asarray(im.convert("L").resize((size + 1, size)), dtype=np.int16)
    except Exception:
        return None
    return np.packbits((a[:, 1:] > a[:, :-1]).ravel())

def _gray32(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((32, 32)), dtype=np.float32).ravel()

def pixel_corr(path_a, path_b):
    a, b = _gray32(path_a), _gray32(path_b)
    if a.std() == 0 or b.std() == 0:
        return float(np.array_equal(a, b))
    return float(np.corrcoef(a, b)[0, 1])

def cross_split_near_dups(df, split_col, max_bits=NEAR_DUP_BITS, min_corr=DUP_MIN_CORR, chunk=400):
    # Stage 1: dHash finds candidates in ANY other split. Stage 2: pixel correlation confirms real duplicates
    # (low-texture images such as closed eyes can share a hash without being the same image).
    df = df[df["hash"].notna()].reset_index(drop=True)
    H, S = np.stack(df["hash"].values), df[split_col].astype(str).values
    summary, pairs = [], []
    for a in pd.unique(S):
        ia, ib = np.where(S == a)[0], np.where(S != a)[0]
        best_d, best_j = np.full(len(ia), 65), np.zeros(len(ia), dtype=int)
        for s0 in range(0, len(ib), chunk):
            blk = ib[s0:s0 + chunk]
            d = _POPCOUNT[np.bitwise_xor(H[ia][:, None, :], H[blk][None, :, :])].sum(axis=2)
            j = d.argmin(axis=1); dmin = d[np.arange(len(ia)), j]
            better = dmin < best_d
            best_d[better], best_j[better] = dmin[better], blk[j[better]]
        cand = np.where(best_d <= max_bits)[0]
        corr = np.array([pixel_corr(df.loc[ia[k], "abs_path"], df.loc[best_j[k], "abs_path"]) for k in cand])
        confirmed = int((corr >= min_corr).sum()) if len(cand) else 0
        summary.append(dict(split=a, images=len(ia), hash_candidates=len(cand), confirmed_duplicates=confirmed,
                            share_confirmed=confirmed / len(ia) if len(ia) else np.nan,
                            max_candidate_corr=float(corr.max()) if len(cand) else np.nan))
        for k, c in sorted(zip(cand, corr), key=lambda t: -t[1])[:10]:
            pairs.append(dict(split_a=a, split_b=S[best_j[k]], path_a=df.loc[ia[k], "abs_path"],
                              path_b=df.loc[best_j[k], "abs_path"], bits=int(best_d[k]), corr=round(c, 3),
                              confirmed=bool(c >= min_corr)))
    pairs = pd.DataFrame(pairs)
    if len(pairs):
        pairs = pairs.sort_values("corr", ascending=False).reset_index(drop=True)
    return pd.DataFrame(summary), pairs

def show_pairs(pairs, title, name, n=4):
    from PIL import Image
    pairs = pairs.head(n)
    if pairs.empty:
        return
    fig, axes = plt.subplots(len(pairs), 2, figsize=(5, 2.4 * len(pairs)), squeeze=False)
    for r, row in enumerate(pairs.itertuples()):
        for c, (p, s) in enumerate([(row.path_a, row.split_a), (row.path_b, row.split_b)]):
            with Image.open(p) as im:
                axes[r, c].imshow(im.convert("RGB"))
            axes[r, c].set_title(f"{s}" + (f"  (Δ{row.bits} bits, r={row.corr:.2f})" if c else ""), fontsize=8)
            axes[r, c].axis("off")
    fig.suptitle(title, fontweight="bold")
    save_fig(fig, name)

def show_grid(df, group_col, n_per_group, title, name, img_col="abs_path", cmap=None):
    from PIL import Image
    groups = list(df[group_col].dropna().unique())
    if not groups:
        print("nothing to show"); return
    fig, axes = plt.subplots(len(groups), n_per_group, figsize=(1.9 * n_per_group, 2.0 * len(groups)), squeeze=False)
    for i, g in enumerate(groups):
        sub = df[df[group_col] == g].head(n_per_group)
        for j in range(n_per_group):
            ax = axes[i, j]; ax.axis("off")
            if j < len(sub):
                with Image.open(sub.iloc[j][img_col]) as im:
                    ax.imshow(im.convert("RGB") if cmap is None else im.convert("L"), cmap=cmap)
            if j == 0:
                ax.set_title(str(g), loc="left", fontsize=9)
    fig.suptitle(title, fontweight="bold")
    save_fig(fig, name)
""")

# ---------------------------------------------------------------------------
md(r"""
---
## 1. Students Attention Detection — V3
Features were extracted from webcam video. **label 0 = attentive, 1 = inattentive** (dataset description, Mendeley V3). The cleaning stage added `row_id` and a **70/15/15 split by row order**.
### 1.1 Structure and integrity
""")

code(r"""
att = data["attention_v3"].copy()
LABEL_NAMES = {0: "attentive", 1: "inattentive"}
att["label_name"] = att["label"].map(LABEL_NAMES)
ATT_SPLITS = ["train", "val", "test"]

feature_cols = [c for c in att.columns if c not in ("row_id", "label", "label_name", "split")]
num_cols = att[feature_cols].select_dtypes("number").columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]
bin_cols = [c for c in num_cols if set(att[c].dropna().unique()) <= {0, 1}]
cont_cols = [c for c in num_cols if c not in bin_cols and att[c].nunique() > 10]
count_cols = [c for c in num_cols if c not in bin_cols + cont_cols]

print(f"rows {len(att):,} | features {len(feature_cols)} "
      f"(continuous {len(cont_cols)}, binary {len(bin_cols)}, counts {len(count_cols)}, categorical {len(cat_cols)})")
print("missing values:", int(att[feature_cols].isna().sum().sum()))
dup_feat = att.duplicated(subset=feature_cols, keep=False)
dup_conflict = att[dup_feat].groupby(feature_cols, dropna=False)["label"].nunique().gt(1).sum()
print(f"exact duplicate feature rows: {int(att.duplicated(subset=feature_cols).sum())} "
      f"| duplicate groups with conflicting labels: {int(dup_conflict)}")

profile = pd.DataFrame({
    "dtype": att[feature_cols].dtypes.astype(str),
    "n_unique": att[feature_cols].nunique(),
    "share_zero": [(att[c] == 0).mean() if c in num_cols else np.nan for c in feature_cols],
    "min": [att[c].min() if c in num_cols else np.nan for c in feature_cols],
    "max": [att[c].max() if c in num_cols else np.nan for c in feature_cols],
})
if dup_conflict:
    add_finding("att_dups", "Attention V3",
                f"{int(att.duplicated(subset=feature_cols).sum())} exact duplicate feature rows; {int(dup_conflict)} group(s) carry different labels.",
                "Drop the conflicting duplicates before training (identical input, contradictory target).")
profile["group"] = ["continuous" if c in cont_cols else "binary" if c in bin_cols else
                    "count" if c in count_cols else "categorical" for c in feature_cols]
display(profile)
""")

md(r"""
### 1.2 Target balance and split hygiene
""")

code(r"""
ct = pd.crosstab(att["split"], att["label_name"]).reindex(ATT_SPLITS)
chi2, p_split, _, _ = stats.chi2_contingency(ct)
share_inatt = (ct["inattentive"] / ct.sum(axis=1))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
counts = att["label_name"].value_counts().reindex(["attentive", "inattentive"])
axes[0].bar(counts.index, counts.values, color=["#4C9F70", "#D1495B"])
for i, v in enumerate(counts.values):
    axes[0].text(i, v, f"{v:,}\n({pct(v / counts.sum())})", ha="center", va="bottom", fontsize=9)
axes[0].set_title("Label counts (all rows)"); axes[0].set_ylim(0, counts.max() * 1.25)
(ct.div(ct.sum(axis=1), axis=0) * 100).plot(kind="bar", stacked=True, ax=axes[1], color=["#4C9F70", "#D1495B"], rot=0)
axes[1].set_title(f"Label share per split (χ² p = {p_split:.1e})"); axes[1].set_ylabel("% of rows")
axes[1].legend(title=None, loc="lower right")
save_fig(fig, "att_label_balance")

display(ct.assign(rows=ct.sum(axis=1), inattentive_share=share_inatt.round(3)))
add_finding("att_balance", "Attention V3",
            f"Overall classes are close to balanced (minority/majority = {counts.min() / counts.max():.2f}).",
            "No resampling needed at dataset level; still report balanced accuracy / F1.",
            f"counts {counts.to_dict()}")
if p_split < 0.05:
    add_finding("att_split_drift", "Attention V3",
                f"Inattentive share differs between splits: "
                + ", ".join(f"{s} {pct(v)}" for s, v in share_inatt.items()) + f" (χ² p = {p_split:.1e}).",
                "Re-split so every split has the same label share (block-wise stratified split, see 1.3); "
                "otherwise validation/test scores are not comparable to training.",
                "row-order 70/15/15 split from the cleaning stage")
""")

md(r"""
### 1.3 Row order — are rows consecutive video frames?
The dataset has no subject or video ID column. If neighbouring rows are near-identical, a **random** shuffle split would put almost the same frame in train and test, which is a form of leakage.
""")

code(r"""
runs = (att["label"] != att["label"].shift()).cumsum()
run_len = att.groupby(runs).size()
sim_cols = [c for c in ["face_x", "face_y", "face_w", "head_yaw", "head_pitch", "nose_tip_x"] if c in att]
consecutive = att[sim_cols].diff().abs().median()
shuffled = (att[sim_cols].reset_index(drop=True) - att[sim_cols].sample(frac=1, random_state=SEED).reset_index(drop=True)).abs().median()
sim = pd.DataFrame({"median |Δ| consecutive rows": consecutive, "median |Δ| random pairs": shuffled})
sim["ratio"] = sim.iloc[:, 0] / sim.iloc[:, 1].replace(0, np.nan)
display(sim)
print(f"label runs: {len(run_len):,} | mean run length {run_len.mean():.1f} rows | longest {run_len.max()} rows")

fig, ax = plt.subplots(figsize=(11, 3.2))
ax.plot(att["row_id"], att["label"].rolling(100, center=True).mean(), color="#D1495B", lw=1.2)
for s in ATT_SPLITS[1:]:
    ax.axvline(att.loc[att["split"] == s, "row_id"].min(), color="k", ls="--", lw=1)
    ax.text(att.loc[att["split"] == s, "row_id"].min(), 1.02, s, fontsize=9)
ax.set_ylim(0, 1.1); ax.set_xlabel("row_id (file order)"); ax.set_ylabel("inattentive share (rolling 100)")
ax.set_title("Label share along the file order, with the cleaning-stage split boundaries")
save_fig(fig, "att_row_order")

if sim["ratio"].median() < 0.5:
    add_finding("att_row_order", "Attention V3",
                f"Neighbouring rows are much more similar than random pairs (median ratio {sim['ratio'].median():.2f}); "
                f"labels come in runs (mean {run_len.mean():.1f} rows): rows behave like consecutive frames.",
                "Never shuffle rows randomly before splitting. Split by contiguous blocks "
                "(e.g. GroupKFold on label runs / blocks of rows) so near-duplicate frames stay together.",
                "no subject/video id available in V3")
""")

md(r"""
### 1.4 Zeros that mean "absent"
When the phone, a hand or the face is not detected, the coordinate columns are filled with **0**. Treating 0 as a real coordinate would distort the means and distributions, so the checks below confirm the rule. A copy of the data (`att_na`) is then made where those zeros become `NaN`.
""")

code(r"""
RULES = [
    ("phone_present == 0", att["phone_present"] == 0, ["phone_loc_x", "phone_loc_y", "phone_conf"]),
    ("hand_count == 0",    att["hand_count"] == 0,    ["left_hand_x", "left_hand_y", "right_hand_x", "right_hand_y"]),
    ("face_present == 0",  att["face_present"] == 0,  ["face_x", "face_y", "face_w", "face_h", "face_conf",
                                                       "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y",
                                                       "nose_tip_x", "nose_tip_y", "mouth_x", "mouth_y"]),
    ("gaze_direction == 'none'", att["gaze_direction"] == "none", ["gazePoint_x", "gazePoint_y", "pupil_left_x",
                                                                    "pupil_left_y", "pupil_right_x", "pupil_right_y"]),
]
rows = []
for name, mask, cols in RULES:
    cols = [c for c in cols if c in att]
    rows.append(dict(condition=name, rows=int(mask.sum()), columns=", ".join(cols),
                     share_all_zero=(att.loc[mask, cols] == 0).all(axis=1).mean() if mask.any() else np.nan,
                     zero_when_present=(att.loc[~mask, cols] == 0).all(axis=1).mean()))
absent = pd.DataFrame(rows)
display(absent)

header("gaze_direction vs gaze_on_screen (row counts)")
gz = pd.crosstab(att["gaze_direction"], att["gaze_on_screen"])
display(gz.assign(All=gz.sum(axis=1)))
if (gz.gt(0).sum(axis=1) == 1).all():
    on = gz.index[gz.get(1, pd.Series(0, index=gz.index)) > 0].tolist()
    add_finding("att_gaze_redundant", "Attention V3",
                f"gaze_on_screen is fully determined by gaze_direction (on-screen ⇔ {on}).",
                "Keep only one of the two in linear models (they carry the same information); "
                "'none' (no gaze detected) should be its own category, not 'off-screen'.")

att_na = att.copy()
for name, mask, cols in RULES:
    cols = [c for c in cols if c in att_na]
    att_na.loc[mask, cols] = att_na.loc[mask, cols].replace(0, np.nan)
for side in ["left", "right"]:   # one detected hand: the other hand's coordinates are 0
    cols = [f"{side}_hand_x", f"{side}_hand_y"]
    if all(c in att_na for c in cols):
        both0 = (att_na[cols] == 0).all(axis=1)
        att_na.loc[both0, cols] = np.nan

add_finding("att_zero_absent", "Attention V3",
            "Missing detections are encoded as 0 (phone / hand / face / gaze coordinates), not as NaN.",
            "Treat these zeros as 'absent': keep the presence flag as its own feature and impute or mask the "
            "coordinates, otherwise scaling and distance features are distorted.",
            "; ".join(f"{r.condition}: {pct(r.share_all_zero)} all-zero" for r in absent.itertuples()))
""")

md(r"""
### 1.5 Which features separate the two classes?
Single-feature ROC-AUC is computed on the **train split only**. AUC = 0.5 means no signal and 1.0 means the feature alone separates the classes. Categorical levels are one-hot encoded. A **depth-3 decision tree** then shows how much simple rules can explain. It is not the final model, only a check of the data.
""")

code(r"""
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text

X_all = pd.get_dummies(att[feature_cols], columns=cat_cols, dtype=int)
y_all = att["label"]
tr = att["split"] == "train"

auc = {}
for c in X_all.columns:
    x = X_all.loc[tr, c]
    if x.nunique() > 1:
        a = roc_auc_score(y_all[tr], x)
        auc[c] = max(a, 1 - a)
auc = pd.Series(auc).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
top = auc.head(15)[::-1]
ax.barh(top.index, top.values, color=["#D1495B" if v >= 0.9 else "#4C72B0" for v in top.values])
ax.axvline(0.5, color="grey", ls=":"); ax.axvline(0.9, color="#D1495B", ls="--", lw=1)
ax.set_xlim(0.5, 1); ax.set_xlabel("single-feature ROC-AUC (train split, direction-free)")
ax.set_title("Top 15 features by individual class separation")
save_fig(fig, "att_feature_auc")

tree = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(X_all[tr], y_all[tr])
scores = {s: balanced_accuracy_score(y_all[att["split"] == s], tree.predict(X_all[att["split"] == s])) for s in ATT_SPLITS}
print("depth-3 tree balanced accuracy:", {k: round(v, 3) for k, v in scores.items()})
print(export_text(tree, feature_names=list(X_all.columns), max_depth=3))

strong = auc[auc >= 0.9]
add_finding("att_signal", "Attention V3",
            f"Strongest single features: {', '.join(f'{k} ({v:.2f})' for k, v in auc.head(3).items())}; "
            f"a depth-3 tree reaches balanced accuracy {scores['test']:.2f} on test.",
            "Gaze/head-pose/phone features carry most of the signal. Use the tree as the baseline any model must beat"
            + ("; features with AUC ≥ 0.9 may be part of how labels were produced — check the labelling "
               "protocol and test the model without them (leakage check)." if len(strong) else "."),
            f"{len(strong)} feature(s) with AUC ≥ 0.9")
""")

md(r"""
### 1.6 Distributions of the most informative features
These plots use `att_na`, so "absent" zeros are excluded. Binary flags are shown as rates.
""")

code(r"""
top_cont = [c for c in auc.index if c in cont_cols][:9]
fig, axes = plt.subplots(3, 3, figsize=(12, 8.5))
for ax, c in zip(axes.ravel(), top_cont):
    sns.histplot(data=att_na, x=c, hue="label_name", stat="density", common_norm=False, bins=40,
                 element="step", palette={"attentive": "#4C9F70", "inattentive": "#D1495B"}, ax=ax, legend=ax is axes[0, 0])
    ax.set_title(f"{c} (AUC {auc[c]:.2f})"); ax.set_xlabel("")
for ax in axes.ravel()[len(top_cont):]:
    ax.axis("off")
fig.suptitle("Continuous features by class (absent detections removed)", fontweight="bold")
save_fig(fig, "att_top_distributions")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1.3, 1, 1.2]})
rates = att.groupby("label_name")[bin_cols].mean().T
rates.plot(kind="barh", ax=axes[0], color=["#4C9F70", "#D1495B"])
axes[0].set_title("Binary flags: share of rows = 1"); axes[0].set_xlabel("share"); axes[0].legend(title=None)
for ax, c in zip(axes[1:], cat_cols[:2]):
    tab = pd.crosstab(att[c], att["label"])
    rate = (tab[1] / tab.sum(axis=1)).sort_values()
    ax.barh([f"{i} (n={tab.loc[i].sum():,})" for i in rate.index], rate.values, color="#D1495B")
    ax.set_xlim(0, 1); ax.set_title(f"P(inattentive | {c})")
save_fig(fig, "att_flags_categories")
""")

md(r"""
### 1.7 Ranges, outliers and redundancy
""")

code(r"""
desc = att_na[cont_cols].describe().T
q1, q3 = att_na[cont_cols].quantile(0.25), att_na[cont_cols].quantile(0.75)
iqr = q3 - q1
desc["outlier_share_IQR"] = ((att_na[cont_cols] < q1 - 1.5 * iqr) | (att_na[cont_cols] > q3 + 1.5 * iqr)).mean()
desc["missing_after_absent"] = att_na[cont_cols].isna().mean()
display(desc.sort_values("outlier_share_IQR", ascending=False)[["count", "mean", "std", "min", "50%", "max",
                                                                 "outlier_share_IQR", "missing_after_absent"]])

corr = att_na[cont_cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.6}, ax=ax,
            xticklabels=True, yticklabels=True)
ax.set_title("Spearman correlation — continuous features (absent detections removed)")
ax.tick_params(labelsize=7)
save_fig(fig, "att_correlation")

upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
redundant = upper[upper.abs() >= 0.9].sort_values(key=np.abs, ascending=False)
display(redundant.rename("spearman_r").to_frame().head(20))
if len(redundant):
    add_finding("att_redundant", "Attention V3",
                f"{len(redundant)} feature pairs have |Spearman r| ≥ 0.9 (mostly facial landmark coordinates that move together).",
                "Reduce redundancy: express landmarks relative to the face box (or keep one per group) before training "
                "linear / distance-based models; tree models are less affected.",
                ", ".join(f"{a}~{b}" for (a, b) in redundant.index[:4]))
hi_out = desc["outlier_share_IQR"].sort_values(ascending=False).head(3)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
sns.histplot(att_na["face_x"].dropna(), bins=60, ax=axes[0], color="#4C72B0")
axes[0].set_title("face_x — several position clusters")
sns.scatterplot(data=att_na.sample(2000, random_state=SEED), x="face_x", y="face_y", hue="label_name", s=8,
                palette={"attentive": "#4C9F70", "inattentive": "#D1495B"}, ax=axes[1])
axes[1].invert_yaxis(); axes[1].set_title("Face position in the frame (sample of 2,000)")
save_fig(fig, "att_position_clusters")
add_finding("att_outliers", "Attention V3",
            "High IQR-'outlier' shares (" + ", ".join(f"{k} {pct(v)}" for k, v in hi_out.items())
            + ") come from separate clusters of face position in the frame (different seats / camera set-ups), not from errors.",
            "Do not delete these rows. Normalise landmarks relative to the face box (or frame size) so the model learns "
            "posture, not where the student sat; use robust scaling.")
""")

md(r"""
### 1.8 The whole feature space in 2-D (PCA and t-SNE)
These checks look at whether the two classes form separate groups when all features are used together, and whether the splits cover the same region. **Neighbour agreement** is the share of a row's 5 nearest neighbours that have the same label. It is computed twice:
* with all neighbours;
* excluding rows within ±10 positions in the file, which removes the near-copies from consecutive frames.
""")

code(r"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

X_emb = pd.get_dummies(att_na[feature_cols], columns=cat_cols, dtype=float)
for c in X_emb.columns[X_emb.isna().any()].tolist():
    X_emb[c + "_absent"] = X_emb[c].isna().astype(float)
X_emb = X_emb.fillna(X_emb.median())
X_std = StandardScaler().fit_transform(X_emb)

pca = PCA(random_state=SEED).fit(X_std)
cum = np.cumsum(pca.explained_variance_ratio_)
n90 = int(np.searchsorted(cum, 0.90) + 1)
pc = pca.transform(X_std)[:, :2]
print(f"{X_std.shape[1]} standardised inputs | PC1+PC2 explain {pct(cum[1])} | {n90} components reach 90% of the variance")

emb_idx = np.sort(rng.choice(len(att), size=min(2500, len(att)), replace=False))
ts = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(X_std[emb_idx])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
lab_pal = {"attentive": "#4C9F70", "inattentive": "#D1495B"}
sns.scatterplot(x=pc[emb_idx, 0], y=pc[emb_idx, 1], hue=att["label_name"].values[emb_idx], s=8, palette=lab_pal, ax=axes[0])
axes[0].set_title("PCA (PC1 vs PC2) — by label"); axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
sns.scatterplot(x=ts[:, 0], y=ts[:, 1], hue=att["label_name"].values[emb_idx], s=8, palette=lab_pal, ax=axes[1])
axes[1].set_title("t-SNE — by label")
sns.scatterplot(x=ts[:, 0], y=ts[:, 1], hue=att["split"].values[emb_idx], hue_order=ATT_SPLITS, s=8, ax=axes[2])
axes[2].set_title("t-SNE — by split")
for ax in axes[1:]:
    ax.set_xticks([]); ax.set_yticks([])
save_fig(fig, "att_embedding")

nn = NearestNeighbors(n_neighbors=26).fit(X_std)
_, idx = nn.kneighbors(X_std)
idx = idx[:, 1:]
rows = att["row_id"].to_numpy()
y = att["label"].to_numpy()
near_in_file = np.abs(rows[idx] - rows[:, None]) <= 10
agree_all = (y[idx[:, :5]] == y[:, None]).mean()
far = [r[~m][:5] for r, m in zip(idx, near_in_file)]
agree_far = np.mean([(y[f] == y[i]).mean() for i, f in enumerate(far) if len(f)])
print(f"nearest neighbour lies within ±10 rows in the file: {pct(near_in_file[:, 0].mean())}")
print(f"5-NN label agreement: all neighbours {pct(agree_all)} | excluding ±10-row neighbours {pct(agree_far)}")

far_pc = pc[:, 0] > np.percentile(pc[:, 0], 99.5) * 0.5 + np.median(pc[:, 0]) * 0.5
if far_pc.any():
    print(f"far-right PCA group: {int(far_pc.sum())} rows, of which face_present == 0: {pct((att.loc[far_pc, 'face_present'] == 0).mean())}")

sp = att["split"].to_numpy()
same_split = (sp[idx[:, :5]] == sp[:, None]).mean(axis=1)
split_cohesion = pd.Series(same_split).groupby(sp).mean().reindex(ATT_SPLITS)
expected = att["split"].value_counts(normalize=True).reindex(ATT_SPLITS)
header("Share of each row's 5 nearest neighbours that come from the same split (vs. share expected by chance)")
display(pd.DataFrame({"same_split_neighbours": split_cohesion, "expected_if_mixed": expected}))
if (split_cohesion["test"] > 2 * expected["test"]) or (split_cohesion["val"] > 2 * expected["val"]):
    add_finding("att_split_regions", "Attention V3",
                f"Val/test rows sit in their own regions of feature space ({pct(split_cohesion['val'])} / {pct(split_cohesion['test'])} "
                f"of their neighbours are from the same split, vs {pct(expected['val'])} / {pct(expected['test'])} if mixed).",
                "The row-order split puts different recording sessions in val/test. That is a fair test of generalisation to new "
                "sessions, but it also explains the label-share drift; keep a session-like block split and report both "
                "in-session and cross-session scores.")

drop = agree_all - agree_far
add_finding("att_embedding", "Attention V3",
            f"5-NN label agreement is {pct(agree_all)} with all neighbours and {pct(agree_far)} when neighbouring frames (±10 rows) "
            f"are excluded; {pct(near_in_file[:, 0].mean())} of rows have their nearest neighbour within ±10 rows.",
            ("The classes stay separable beyond near-copies of the same moment, so the signal is real; "
             if drop < 0.05 else
             "Much of the apparent separability comes from near-copies of the same moment; validate with block-wise splits. ")
            + f"{n90} principal components hold 90% of the variance of {len(feature_cols)} features → strong redundancy, "
              "so a compact feature set (or tree model) is enough.")
""")

# ---------------------------------------------------------------------------
md(r"""
---
## 2. DAiSEE — engagement, boredom, confusion, frustration
The cleaning stage took **10 frames per clip** (1 fps, 224×224). It kept the **official person-disjoint split**. Every frame inherits its clip's labels (levels 0 = very low … 3 = very high), so label statistics below are computed **per clip**. Counting frames would multiply the evidence by 10.
### 2.1 Structure and split hygiene
""")

code(r"""
dai = data["daisee"].copy()
AFFECTS = ["boredom", "engagement", "confusion", "frustration"]
D_SPLITS = ["Train", "Validation", "Test"]
dai["person"] = dai["person"].astype(str)
dai["clip"] = dai["clip"].astype(str)

frames_per_clip = dai.groupby("clip").size()
label_varies = dai.groupby("clip")[AFFECTS].nunique().gt(1).any(axis=1).sum()
clips = dai.drop_duplicates("clip")[["split", "person", "clip"] + AFFECTS].reset_index(drop=True)

summary = dai.groupby("split").agg(frames=("path", "size"), clips=("clip", "nunique"), persons=("person", "nunique")).reindex(D_SPLITS)
summary["clips_per_person"] = (summary["clips"] / summary["persons"]).round(1)
display(summary)
print("frames per clip:", frames_per_clip.value_counts().to_dict())
print("clips whose label changes between frames:", int(label_varies))
print("persons in more than one split:", split_overlap(dai, "split", "person"))
print("clips in more than one split:", split_overlap(dai, "split", "clip"))
print("label values outside 0-3:", int((~dai[AFFECTS].isin([0, 1, 2, 3])).any(axis=1).sum()))

add_finding("dai_units", "DAiSEE",
            f"{len(dai):,} frames come from {len(clips):,} clips ({frames_per_clip.median():.0f} frames each) and share the clip label.",
            "Treat the clip (or the person) as the unit: aggregate frame predictions per clip, and never split frames of one clip "
            "across train/test. Effective sample size = clips, not frames.")
if split_overlap(dai, "split", "person") == 0:
    add_finding("dai_split_ok", "DAiSEE", "No person appears in more than one split (official split kept).",
                "Keep this person-disjoint split for all experiments.")
""")

md(r"""
### 2.2 Label distributions (per clip)
Each 10-second clip got **one ordinal judgement per state** (0–3). These are subjective ratings of a whole clip, not measurements. Two things follow from that:
* neighbouring levels (e.g. 2 vs 3) are hard to tell apart from a single frame;
* the rare extreme levels carry the most label noise.

So the distributions below are also used to decide whether a binary target is more trustworthy than 4 levels.
""")

code(r"""
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
level_share = {}
for ax, a in zip(axes.ravel(), AFFECTS):
    tab = pd.crosstab(clips["split"], clips[a], normalize="index").reindex(D_SPLITS) * 100
    tab = tab.reindex(columns=[0, 1, 2, 3], fill_value=0)
    tab.T.plot(kind="bar", ax=ax, rot=0, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_title(a); ax.set_xlabel("level (0 very low … 3 very high)"); ax.set_ylabel("% of clips")
    ax.legend(title=None, fontsize=8)
    level_share[a] = clips[a].value_counts(normalize=True).reindex([0, 1, 2, 3], fill_value=0)
fig.suptitle("DAiSEE label levels per split (share of clips)", fontweight="bold")
save_fig(fig, "dai_label_levels")

share = pd.DataFrame(level_share).T
share.columns = [f"level_{c}" for c in share.columns]
share["rarest_level_share"] = share.min(axis=1)
share["max/min_ratio"] = [imbalance_ratio(clips[a].value_counts().reindex([0, 1, 2, 3], fill_value=0)) for a in AFFECTS]
display(share)

binary = pd.DataFrame({
    "engagement_low (0-1)": clips.groupby("split")["engagement"].apply(lambda s: (s <= 1).mean()),
    "boredom_high (2-3)":   clips.groupby("split")["boredom"].apply(lambda s: (s >= 2).mean()),
    "confusion_any (1-3)":  clips.groupby("split")["confusion"].apply(lambda s: (s >= 1).mean()),
}).reindex(D_SPLITS)
header("Share of the positive class if the labels are binarised")
display(binary)

low = binary["engagement_low (0-1)"]
if low.max() / max(low.min(), 1e-9) > 1.5:
    add_finding("dai_split_shift", "DAiSEE",
                "Low-engagement share differs between the official splits: " + ", ".join(f"{k} {pct(v)}" for k, v in low.items()) + ".",
                "Compare models on the same split and report per-split metrics; tune thresholds on Validation knowing it has more low-engagement clips.")
worst = share["max/min_ratio"].idxmax()
add_finding("dai_imbalance", "DAiSEE",
            f"Strong level imbalance: rarest level of {worst} is only {pct(share.loc[worst, 'rarest_level_share'])} of clips; "
            f"engagement level 0 = {pct(share.loc['engagement', 'level_0'])}.",
            "Do not train a 4-level classifier on raw levels. Binarise (e.g. engagement low 0-1 vs high 2-3, boredom 0-1 vs 2-3) "
            "and use class weights / balanced sampling; report macro-F1.",
            f"low-engagement share per split: {binary['engagement_low (0-1)'].round(3).to_dict()}")
""")

md(r"""
### 2.3 How the four states relate, and how much depends on the person
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
c = clips[AFFECTS].corr(method="spearman")
sns.heatmap(c, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, ax=axes[0], square=True)
axes[0].set_title("Spearman correlation between states (clips)")

person_mean = clips.groupby("person")["engagement"].mean()
between_share = person_mean.var() / clips["engagement"].var()
sns.histplot(person_mean, bins=25, ax=axes[1], color="#4C72B0")
axes[1].set_title(f"Mean engagement per person (n={len(person_mean)})")
axes[1].set_xlabel("mean engagement level of the person's clips")
save_fig(fig, "dai_affect_corr_person")

print(f"share of engagement variance explained by the person: {between_share:.2f}")
pair = c.where(np.triu(np.ones(c.shape, dtype=bool), k=1)).stack().sort_values(key=np.abs, ascending=False)
add_finding("dai_person_effect", "DAiSEE",
            f"About {pct(between_share, 0)} of engagement variance is between persons; strongest state pair: "
            f"{pair.index[0][0]}–{pair.index[0][1]} (r = {pair.iloc[0]:.2f}).",
            "Person identity is a confounder: keep person-disjoint splits and consider per-user calibration "
            "(baseline from the first minutes of a session) in Maslak.")
""")

md(r"""
### 2.4 Frames (needs `daisee_frames.zip`)
""")

code(r"""
dai_root = prepare_images("daisee")
dai_img, dai_abs = None, None
if dai_root is not None:
    dai_abs = resolve_paths(dai["path"], dai_root)
    if dai_abs is not None:
        dai_img = sample_existing(dai, ["split", "engagement"], max(IMG_PER_GROUP // 4, 10), dai_abs)
        print(f"sampled {len(dai_img)} frames")
if dai_img is not None and len(dai_img):
    st = pd.concat([dai_img.reset_index(drop=True), image_stats(dai_img["abs_path"])], axis=1)
    print("unreadable:", int((~st["readable"]).sum()), "| sizes:", st.groupby(["width", "height"]).size().to_dict())
    display(st.groupby("split")[["brightness", "contrast", "sharpness"]].median().reindex(D_SPLITS))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, m in zip(axes, ["brightness", "contrast", "sharpness"]):
        sns.boxplot(data=st, x="engagement", y=m, ax=ax, color="#8DB3E2", showfliers=False)
        ax.set_title(f"{m} by engagement level")
    save_fig(fig, "dai_image_quality")

    fc = face_counts(st["abs_path"])
    if fc is not None:
        st["face_found"] = fc > 0
        header("Share of sampled frames where a frontal face is detected (OpenCV face detector)")
        display(pd.concat([st.groupby("split")["face_found"].mean().reindex(D_SPLITS).rename("by split"),
                           st.groupby("engagement")["face_found"].mean().rename("by engagement level")], axis=1).T)
        by_eng = st.groupby("engagement")["face_found"].mean()
        add_finding("dai_faces", "DAiSEE",
                    f"A frontal face is detected in {pct(st['face_found'].mean())} of sampled frames "
                    f"(lowest engagement-level rate: level {by_eng.idxmin()} = {pct(by_eng.min())}).",
                    "Maslak must treat 'no face found' as its own state (away / turned / camera blocked), not silently as "
                    "inattentive, and log the face-detection confidence so low-quality frames can be filtered in the real pipeline.")

    # near-duplicate check inside clips (consecutive frames vs frames from different clips)
    sample_clips = dai_img["clip"].drop_duplicates().head(40)
    same, other = [], []
    paths_by_clip = {k: sorted(v) for k, v in dai[dai["clip"].isin(sample_clips)].groupby("clip")["path"]}
    clip_list = list(paths_by_clip)
    for i, k in enumerate(clip_list):
        p = [dai_abs(x) for x in paths_by_clip[k] if dai_abs(x).exists()]
        q = [dai_abs(x) for x in paths_by_clip[clip_list[(i + 1) % len(clip_list)]] if dai_abs(x).exists()]
        if len(p) >= 2:
            same.append(frame_diff(p[0], p[1]))
        if p and q:
            other.append(frame_diff(p[0], q[0]))
    if same and other:
        print(f"mean pixel difference: consecutive frames {np.median(same):.1f} vs different clips {np.median(other):.1f} (0-255)")
        add_finding("dai_near_dup", "DAiSEE",
                    f"Consecutive frames of a clip are near-duplicates (median pixel diff {np.median(same):.1f} vs {np.median(other):.1f} across clips).",
                    "Frame-level augmentation or a temporal model (sequence of frames) adds more than extra frames from the same clip.")
    show_grid(dai_img[dai_img["split"] == "Train"].sort_values("engagement"), "engagement", 6,
              "DAiSEE sample frames by engagement level (Train)", "dai_samples")
""")

# ---------------------------------------------------------------------------
md(r"""
---
## 3. UTA-RLDD — drowsiness (alert / low vigilant / drowsy)
Each subject recorded one video per state. The cleaning stage extracted **1 frame per second** (224×224, padded). Only **publishable** subjects (`publishable = True`) may appear in figures or reports.
### 3.1 Structure and split hygiene
""")

code(r"""
rl = data["rldd"].copy()
CLASS_ORDER = ["alert", "low_vigilant", "drowsy"]
rl["class_name"] = pd.Categorical(rl["class_name"], CLASS_ORDER, ordered=True)
rl["video"] = rl["fold"].astype(str) + "/s" + rl["subject"].astype(str) + "/c" + rl["label"].astype(str)

videos = rl.groupby(["video", "fold", "fold_num", "subject", "class_name"], observed=True).agg(
    frames=("path", "size"), publishable=("publishable", "first")).reset_index()
videos["minutes"] = videos["frames"] / 60   # 1 frame per second

print(f"frames {len(rl):,} | videos {len(videos)} | subjects {rl['subject'].nunique()} | folds {rl['fold'].nunique()}")
print("subjects in more than one fold number:", split_overlap(rl, "fold_num", "subject"))
per_subj = rl.groupby("subject")["class_name"].nunique()
print("subjects without all 3 states:", per_subj[per_subj < 3].index.tolist())
print("frames with binary = NaN (low_vigilant):", int(rl["binary"].isna().sum()))
display(rl.groupby("fold_num").agg(subjects=("subject", "nunique"), frames=("path", "size"),
                                   publishable_share=("publishable", "mean")))
display(videos.groupby("class_name", observed=True)["minutes"].describe())

fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
cls_counts = rl["class_name"].value_counts().reindex(CLASS_ORDER)
axes[0].bar(CLASS_ORDER, cls_counts.values, color=["#4C9F70", "#E1A940", "#D1495B"])
for i, v in enumerate(cls_counts.values):
    axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
axes[0].set_title("Frames per state")
sns.histplot(data=videos, x="minutes", hue="class_name", bins=25, multiple="stack", ax=axes[1],
             palette=["#4C9F70", "#E1A940", "#D1495B"])
axes[1].set_title("Video length (minutes) per state")
save_fig(fig, "rldd_balance")

heat = rl.pivot_table(index="subject", columns="class_name", values="path", aggfunc="size", observed=False).fillna(0) / 60
fig, ax = plt.subplots(figsize=(13, 3.2))
sns.heatmap(heat.T, cmap="mako_r", cbar_kws={"label": "minutes"}, ax=ax)
ax.set_title("Minutes of video per subject and state (0 = missing video)"); ax.set_xlabel("subject"); ax.set_ylabel("")
ax.tick_params(axis="y", rotation=0)
save_fig(fig, "rldd_subject_minutes")

add_finding("rldd_balance", "UTA-RLDD",
            f"States are balanced (max/min = {imbalance_ratio(cls_counts):.2f}); {int((per_subj == 3).sum())} of {len(per_subj)} subjects "
            f"have all three states" + (f" (incomplete: {per_subj[per_subj < 3].index.tolist()})." if (per_subj < 3).any() else "."),
            "No resampling needed; evaluate with subject-wise cross-validation over the 5 folds "
            "(fold folders are subject-disjoint)." if split_overlap(rl, "fold_num", "subject") == 0 else
            "Check fold assignment: some subjects appear in several folds.")
add_finding("rldd_middle", "UTA-RLDD",
            "'low_vigilant' is the in-between state and has no binary label in the cleaned table.",
            "Start with binary alert vs drowsy (column `binary`), then try 3 classes; expect most confusion around low_vigilant.")
add_finding("rldd_publish", "UTA-RLDD",
            f"Only {pct(rl['publishable'].mean())} of frames come from subjects who allowed publication.",
            "Use all subjects for training, but show only publishable subjects in reports, slides and GitHub.")
""")

md(r"""
### 3.2 Frames (needs `rldd_frames.zip`)
""")

code(r"""
rl_root = prepare_images("rldd")
rl_img, rl_abs = None, None
if rl_root is not None:
    rl_abs = resolve_paths(rl["path"], rl_root)
    if rl_abs is not None:
        rl_img = sample_existing(rl, ["class_name"], IMG_PER_GROUP, rl_abs)
        print(f"sampled {len(rl_img)} frames")
if rl_img is not None and len(rl_img):
    st = pd.concat([rl_img.reset_index(drop=True), image_stats(rl_img["abs_path"])], axis=1)
    print("unreadable:", int((~st["readable"]).sum()), "| sizes:", st.groupby(["width", "height"]).size().to_dict())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, m in zip(axes, ["brightness", "contrast", "sharpness"]):
        sns.boxplot(data=st, x="class_name", y=m, order=CLASS_ORDER, ax=ax, color="#8DB3E2", showfliers=False)
        ax.set_title(f"{m} by state"); ax.set_xlabel("")
    save_fig(fig, "rldd_image_quality")
    fc = face_counts(st["abs_path"])
    if fc is not None:
        st["face_found"] = fc > 0
        rate = st.groupby("class_name", observed=True)["face_found"].mean()
        header("Share of sampled frames where a frontal face is detected (OpenCV face detector)")
        display(rate.to_frame("face_found_share").T)
        add_finding("rldd_faces", "UTA-RLDD",
                    "Frontal-face detection rate by state: " + ", ".join(f"{k} {pct(v)}" for k, v in rate.items()) + ".",
                    "If detection drops for drowsy frames (head tilted / dropped), a missing face is itself a drowsiness cue: "
                    "log it as a feature instead of discarding those frames.")
    med = st.groupby("class_name", observed=True)["brightness"].median()
    if med.max() - med.min() > 15:
        add_finding("rldd_light", "UTA-RLDD",
                    f"Median brightness differs between states ({med.round(0).to_dict()}).",
                    "Recording conditions may differ per state video: use brightness/contrast augmentation so the model "
                    "does not learn lighting instead of drowsiness.")

    # temporal redundancy: 1 s apart vs 60 s apart within the same video
    diffs = []
    for v in rl_img["video"].drop_duplicates().head(40):
        fr = rl[rl["video"] == v].sort_values("second")
        p = [rl_abs(x) for x in fr["path"].iloc[[0, 1, min(60, len(fr) - 1)]]]
        if all(x.exists() for x in p):
            diffs.append((frame_diff(p[0], p[1]), frame_diff(p[0], p[2])))
    if diffs:
        d1, d60 = np.median([d[0] for d in diffs]), np.median([d[1] for d in diffs])
        print(f"median pixel difference: 1 s apart {d1:.1f} | 60 s apart {d60:.1f}")
        add_finding("rldd_redundant", "UTA-RLDD",
                    f"Frames 1 s apart are nearly identical (pixel diff {d1:.1f} vs {d60:.1f} at 60 s).",
                    "Sub-sample frames (e.g. every 5-10 s) or use short temporal windows; more 1-fps frames add little new information.")

    pub = rl_img[rl_img["publishable"] == True]
    subj = pub.groupby("subject")["class_name"].nunique()
    subj = subj[subj == 3].index[:2]
    grid = pub[pub["subject"].isin(subj)].sort_values(["class_name", "subject"]).groupby("class_name", observed=True).head(4)
    show_grid(grid, "class_name", 4, "RLDD sample frames — publishable subjects only", "rldd_samples")
""")

# ---------------------------------------------------------------------------
md(r"""
---
## 4. MRL Eye — open vs closed eye crops
Each file name encodes the subject, glasses, reflections, lighting and sensor. The cleaning stage removed exact duplicates and images with conflicting labels. It then made a **subject-disjoint 70/15/15 split**.
### 4.1 Structure and split hygiene
""")

code(r"""
mrl = data["mrl"].copy()
MAPS = {
    "eye_open":    {0: "closed", 1: "open"},
    "gender":      {0: "male", 1: "female"},
    "glasses":     {0: "no", 1: "yes"},
    "reflections": {0: "none", 1: "small", 2: "big"},
    "good_light":  {0: "bad", 1: "good"},
    "sensor":      {1: "RealSense 640×480", 2: "IDS 1280×1024", 3: "Aptina 752×480"},
}
for c, m in MAPS.items():
    if c in mrl:
        mrl[c + "_name"] = mrl[c].map(m)
M_SPLITS = ["train", "val", "test"]

print(f"images {len(mrl):,} | subjects {mrl['subject'].nunique()} | subjects in more than one split: {split_overlap(mrl, 'split', 'subject')}")
display(mrl.groupby("split").agg(images=("path", "size"), subjects=("subject", "nunique"),
                                 open_share=("eye_open", "mean"), glasses_share=("glasses", "mean")).reindex(M_SPLITS))

per_subject = mrl.groupby("subject").agg(images=("path", "size"), open_share=("eye_open", "mean")).sort_values("images", ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [2, 1]})
axes[0].bar(per_subject.index, per_subject["images"], color="#4C72B0")
axes[0].tick_params(axis="x", rotation=90, labelsize=7); axes[0].set_title("Images per subject")
sns.histplot(per_subject["open_share"], bins=15, ax=axes[1], color="#55A868")
axes[1].set_title("Open-eye share per subject")
save_fig(fig, "mrl_subjects")

mct = pd.crosstab(mrl["split"], mrl["eye_open"]).reindex(M_SPLITS)
_, p_mrl, _, _ = stats.chi2_contingency(mct)
open_by_split = mrl.groupby("split")["eye_open"].mean().reindex(M_SPLITS)
if open_by_split.max() - open_by_split.min() > 0.05:
    add_finding("mrl_split_drift", "MRL Eye",
                "Open-eye share differs between splits: " + ", ".join(f"{k} {pct(v)}" for k, v in open_by_split.items())
                + f" (χ² p = {p_mrl:.1e}) — a side effect of splitting whole subjects.",
                "Report balanced accuracy / per-class recall, and choose the decision threshold on validation, not on raw accuracy.")
top_share = per_subject["images"].iloc[0] / len(mrl)
add_finding("mrl_balance", "MRL Eye",
            f"Open/closed are balanced overall ({pct(mrl['eye_open'].mean())} open) and the split is subject-disjoint; "
            f"the largest subject holds {pct(top_share)} of all images.",
            "Keep the subject-disjoint split; weight or cap images per subject so a few subjects do not dominate training.")
""")

md(r"""
### 4.2 Capture conditions
""")

code(r"""
conds = ["glasses_name", "reflections_name", "good_light_name", "sensor_name", "gender_name"]
fig, axes = plt.subplots(1, len(conds), figsize=(16, 3.6))
cond_rows = []
for ax, c in zip(axes, conds):
    g = mrl.groupby(c).agg(images=("path", "size"), open_share=("eye_open", "mean"))
    ax.bar(g.index.astype(str), g["images"], color="#8DB3E2")
    ax2 = ax.twinx(); ax2.plot(g.index.astype(str), g["open_share"], "o-", color="#D1495B"); ax2.set_ylim(0, 1)
    ax.set_title(c.replace("_name", "")); ax.tick_params(axis="x", rotation=25, labelsize=8)
    for k, r in g.iterrows():
        cond_rows.append(dict(condition=c.replace("_name", ""), value=k, images=r["images"],
                              share=r["images"] / len(mrl), open_share=r["open_share"]))
axes[0].set_ylabel("images (bars)"); ax2.set_ylabel("open share (red)")
save_fig(fig, "mrl_conditions")
cond_tab = pd.DataFrame(cond_rows)
display(cond_tab)

fig, ax = plt.subplots(figsize=(6, 3.5))
sizes = mrl["width"].clip(upper=mrl["width"].quantile(0.99))
sns.histplot(data=mrl.assign(width_c=sizes), x="width_c", hue="sensor_name", bins=40, multiple="stack", ax=ax)
ax.set_title("Eye-crop width (px) by sensor"); ax.set_xlabel("width (px, clipped at p99)")
save_fig(fig, "mrl_sizes")
print("square crops:", pct((mrl["width"] == mrl["height"]).mean()), "| width range:", mrl["width"].min(), "-", mrl["width"].max())

sens = cond_tab[cond_tab["condition"] == "sensor"].set_index("value")
if sens["open_share"].max() - sens["open_share"].min() > 0.2:
    add_finding("mrl_sensor_confound", "MRL Eye",
                "Eye state is confounded with the sensor: open-eye share " + ", ".join(f"{k} {pct(v)}" for k, v in sens["open_share"].items()) + ".",
                "A model could learn 'which camera' instead of 'eye closed'. Evaluate per sensor and balance open/closed within each sensor when sampling.")
dom = cond_tab[cond_tab["condition"] == "sensor"].sort_values("share", ascending=False).iloc[0]
add_finding("mrl_conditions", "MRL Eye",
            f"Capture conditions are uneven: {dom['value']} provides {pct(dom['share'])} of images; "
            f"glasses {pct(mrl['glasses'].mean())}, bad light {pct(1 - mrl['good_light'].mean())}.",
            "Report accuracy per condition (glasses, reflections, light, sensor) and add blur/brightness augmentation, "
            "because the study-room webcam differs from these near-infrared sensors.")
add_finding("mrl_sizes", "MRL Eye",
            f"Crops vary in size ({mrl['width'].min()}–{mrl['width'].max()} px wide).",
            "Resize every crop to one input size (e.g. 64×64 or 86×86 grayscale) inside the training pipeline.")
""")

md(r"""
### 4.3 Images (downloads the official MRL archive if needed)
""")

code(r"""
mrl_root = prepare_images("mrl")
mrl_img, mrl_abs = None, None
if mrl_root is not None:
    mrl_abs = resolve_paths(mrl["path"], mrl_root)
    if mrl_abs is not None:
        mrl_img = sample_existing(mrl, ["eye_open_name", "good_light_name"], max(IMG_PER_GROUP // 2, 10), mrl_abs)
        print(f"sampled {len(mrl_img)} crops")
if mrl_img is not None and len(mrl_img):
    st = pd.concat([mrl_img.reset_index(drop=True), image_stats(mrl_img["abs_path"])], axis=1)
    print("unreadable:", int((~st["readable"]).sum()), "| modes:", st["mode"].value_counts().to_dict())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, m in zip(axes, ["brightness", "contrast", "sharpness"]):
        sns.boxplot(data=st, x="eye_open_name", y=m, hue="good_light_name", ax=ax, showfliers=False)
        ax.set_title(f"{m}: eye state × light"); ax.set_xlabel("")
    save_fig(fig, "mrl_image_quality")
    show_grid(mrl_img.sort_values(["eye_open_name"]), "eye_open_name", 8, "MRL sample crops", "mrl_samples", cmap="gray")
""")

# ---------------------------------------------------------------------------
md(r"""
---
## 5. FPI-Det — phone and head detection (YOLO format)
There is no CSV for this dataset. The cleaned archive holds `train/val/test` folders with `images/` and `labels/`, one line per box (`class x_center y_center width height`, normalised 0–1). The cleaning stage split multi-box lines, clipped boxes to the image, and removed duplicate and conflicting images.
""")

code(r"""
FPI_CLASS_NAMES = None    # e.g. {0: "head", 1: "phone"}; None = read a .yaml in the archive or infer from box counts
fpi_root = prepare_images("fpi")
fpi_boxes = fpi_images = None
fpi_names_src = None

if fpi_root is not None:
    split_dirs = {}
    for folder, dirs, _ in os.walk(fpi_root):
        if "images" in dirs and "labels" in dirs:
            split_dirs[Path(folder).name] = Path(folder)
        if len(Path(folder).relative_to(fpi_root).parts) > 4:
            dirs[:] = []
    print("splits found:", {k: str(v) for k, v in split_dirs.items()})
    img_rows, box_rows, bad_lines = [], [], 0
    for split, d in split_dirs.items():
        imgs = {p.stem: p for p in (d / "images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        labs = {p.stem: p for p in (d / "labels").glob("*.txt")}
        for stem, p in imgs.items():
            n = 0
            if stem in labs:
                for line in labs[stem].read_text().splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        bad_lines += len(parts) > 0
                        continue
                    c, x, y, w, h = int(float(parts[0])), *map(float, parts[1:])
                    box_rows.append(dict(split=split, image=stem, cls=c, x=x, y=y, w=w, h=h))
                    n += 1
            img_rows.append(dict(split=split, image=stem, path=str(p), has_label_file=stem in labs, boxes=n))
        print(f"{split}: {len(imgs):,} images | {len(labs):,} label files | labels without image: {len(set(labs) - set(imgs))}")
    fpi_images, fpi_boxes = pd.DataFrame(img_rows), pd.DataFrame(box_rows)
    print("malformed label lines:", bad_lines)

if fpi_boxes is not None and len(fpi_boxes):
    if FPI_CLASS_NAMES is None:
        import yaml
        for yml in list(fpi_root.rglob("*.yaml"))[:5]:
            names = (yaml.safe_load(yml.read_text()) or {}).get("names")
            if names:
                FPI_CLASS_NAMES = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
                print("class names from", yml.name, FPI_CLASS_NAMES)
                fpi_names_src = yml.name
                break
    if FPI_CLASS_NAMES is None:
        order = fpi_boxes["cls"].value_counts().index.tolist()
        FPI_CLASS_NAMES = {order[0]: "head", **({order[1]: "phone"} if len(order) > 1 else {})}
        print("no data.yaml found → class names inferred from counts (FPI-Det has more head than phone boxes):", FPI_CLASS_NAMES)
    fpi_boxes["class"] = fpi_boxes["cls"].map(FPI_CLASS_NAMES).fillna(fpi_boxes["cls"].astype(str))
    fpi_boxes["area"] = fpi_boxes["w"] * fpi_boxes["h"]
    if {"head", "phone"} <= set(fpi_boxes["class"]):
        med_area = fpi_boxes.groupby("class")["area"].median()
        ok = med_area["phone"] < med_area["head"]
        print(f"class-name sanity check: median box area head {med_area['head']:.4f} vs phone {med_area['phone']:.4f} → "
              + ("consistent (phones are smaller)" if ok else "SUSPICIOUS — check the class ids with the data owner"))
        add_finding("fpi_class_ids", "FPI-Det",
                    f"Class ids → names: {FPI_CLASS_NAMES} "
                    + ("(sanity check passed: phone boxes are smaller than head boxes)." if ok else "(sanity check FAILED)."),
                    (f"Every training / inference script must use this mapping from {fpi_names_src} "
                     f"({', '.join(f'{k} = {v}' for k, v in sorted(FPI_CLASS_NAMES.items()))}); do not assume another order."
                     if fpi_names_src else
                     "Write a data.yaml with these names next to the cleaned dataset so training scripts use the same mapping."))
    per_img = fpi_boxes.pivot_table(index=["split", "image"], columns="class", values="x", aggfunc="size", fill_value=0)
    fpi_images = fpi_images.merge(per_img.reset_index(), on=["split", "image"], how="left").fillna(0)

    header("Images and boxes per split")
    tab = fpi_images.groupby("split").agg(images=("image", "size"), empty_images=("boxes", lambda s: int((s == 0).sum())),
                                          boxes=("boxes", "sum"))
    for cname in sorted(fpi_boxes["class"].unique()):
        tab[f"{cname}_boxes"] = fpi_images.groupby("split")[cname].sum().astype(int)
        tab[f"images_with_{cname}"] = fpi_images.groupby("split")[cname].apply(lambda s: (s > 0).mean()).round(3)
    tab["empty_share"] = (tab["empty_images"] / tab["images"]).round(3)
    display(tab)
    if "head" in fpi_boxes["class"].values and len(tab) > 1:
        head_cov = tab["images_with_head"]
        if head_cov.max() - head_cov.min() > 0.1 or tab["empty_share"].max() - tab["empty_share"].min() > 0.05:
            odd = head_cov.idxmin()
            phone_only = fpi_images.groupby("split").apply(lambda d: ((d["phone"] > 0) & (d["head"] == 0)).mean())
            many_empty = tab.loc[odd, "empty_share"] > 0.05
            add_finding("fpi_split_shift", "FPI-Det",
                        f"The '{odd}' split differs from the others: {pct(head_cov[odd])} of its images contain a head, "
                        f"{pct(phone_only[odd])} show a phone without any head box and {pct(tab.loc[odd, 'empty_share'])} have no box "
                        "(other splits: " + ", ".join(f"{s} {pct(head_cov[s])} with head / {pct(phone_only[s])} phone-only / "
                                                     f"{pct(tab.loc[s, 'empty_share'])} empty" for s in tab.index if s != odd) + ").",
                        (f"Check with the cleaning stage whether the empty '{odd}' images are true negatives or lost labels; "
                         if many_empty else
                         f"'{odd}' holds more phone-only images (phone in view, head not annotated), so its scores are not "
                         "comparable to test; ")
                        + "report metrics per class and on the test split, not on validation alone.")
""")

code(r"""
if fpi_boxes is not None and len(fpi_boxes):
    classes = sorted(fpi_boxes["class"].unique())
    pal = dict(zip(classes, ["#4C72B0", "#D1495B", "#55A868", "#E1A940"]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    sns.histplot(data=fpi_boxes, x="area", hue="class", log_scale=True, bins=40, element="step",
                 stat="density", common_norm=False, palette=pal, ax=axes[0])
    axes[0].set_title("Box area (share of image, log)")
    counts = fpi_images[classes].clip(upper=6)
    for cname in classes:
        vc = counts[cname].value_counts(normalize=True).sort_index()
        axes[1].plot(vc.index, vc.values, "o-", label=cname, color=pal[cname])
    axes[1].set_title("Boxes per image (6 = 6+)"); axes[1].set_xlabel("boxes"); axes[1].legend()
    for cname in classes:
        sub = fpi_boxes[fpi_boxes["class"] == cname].sample(min(4000, (fpi_boxes["class"] == cname).sum()), random_state=SEED)
        axes[2].scatter(sub["x"], sub["y"], s=2, alpha=0.25, color=pal[cname], label=cname)
    axes[2].set_xlim(0, 1); axes[2].set_ylim(1, 0); axes[2].set_title("Box centres (image coordinates)")
    axes[2].legend(markerscale=5)
    save_fig(fig, "fpi_boxes")

    display(fpi_boxes.groupby("class")[["w", "h", "area"]].describe(percentiles=[0.1, 0.5, 0.9]).T)

    # phone–head proximity: distance from each phone to the nearest head, in head heights
    prox = []
    if {"head", "phone"} <= set(classes):
        for (s, im), g in fpi_boxes.groupby(["split", "image"]):
            heads, phones = g[g["class"] == "head"], g[g["class"] == "phone"]
            if len(heads) and len(phones):
                for ph in phones.itertuples():
                    d = np.hypot(heads["x"] - ph.x, heads["y"] - ph.y) / heads["h"].clip(lower=1e-3)
                    prox.append(d.min())
    if prox:
        prox = pd.Series(prox)
        print(f"phones with a head in the same image: {len(prox):,} | median distance {prox.median():.2f} head-heights "
              f"| within 2 head-heights: {pct((prox <= 2).mean())}")

    from PIL import Image
    pool = fpi_images[fpi_images["phone"] > 0] if "phone" in fpi_images else fpi_images
    sample_ids = pool.sample(min(8, len(pool)), random_state=SEED)
    if len(sample_ids):
        fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
        for ax, r in zip(axes.ravel(), sample_ids.to_dict("records")):
            with Image.open(r["path"]) as im:
                ax.imshow(im.convert("RGB")); W, H = im.size
            img_boxes = fpi_boxes[(fpi_boxes["split"] == r["split"]) & (fpi_boxes["image"] == r["image"])]
            for b in img_boxes.to_dict("records"):
                ax.add_patch(mpatches.Rectangle(((b["x"] - b["w"] / 2) * W, (b["y"] - b["h"] / 2) * H), b["w"] * W, b["h"] * H,
                                                fill=False, lw=1.5, color=pal[b["class"]]))
            ax.axis("off")
        for ax in axes.ravel()[len(sample_ids):]:
            ax.axis("off")
        fig.legend(handles=[mpatches.Patch(color=pal[c], label=c) for c in classes], loc="lower center", ncol=len(classes))
        fig.suptitle("FPI-Det samples with boxes", fontweight="bold")
        save_fig(fig, "fpi_samples")

    sizes = image_stats(fpi_images["path"].sample(min(400, len(fpi_images)), random_state=SEED))
    print("unreadable (sample of 400):", int((~sizes["readable"]).sum()),
          "| most common sizes:", sizes.groupby(["width", "height"]).size().sort_values(ascending=False).head(3).to_dict())

    n_cls = fpi_boxes["class"].value_counts()
    add_finding("fpi_balance", "FPI-Det",
                "Box counts per class: " + ", ".join(f"{k} {v:,}" for k, v in n_cls.items())
                + f" (ratio {imbalance_ratio(n_cls):.1f}); "
                + (f"{pct((fpi_images['phone'] == 0).mean())} of images contain no phone." if "phone" in fpi_images else ""),
                "Keep images without phones (they are the negatives that teach 'no phone'); check per-class mAP, "
                "not only overall mAP.")
    if "phone" in classes:
        ph_area = fpi_boxes.loc[fpi_boxes["class"] == "phone", "area"].median()
        add_finding("fpi_small", "FPI-Det",
                    f"Phones are small objects (median box = {pct(ph_area, 2)} of the image area).",
                    "Train at ≥ 640 px input, keep small-object augmentation (mosaic/scale) and check recall on small phones.")
    if isinstance(prox, pd.Series) and len(prox):
        add_finding("fpi_proximity", "FPI-Det",
                    f"{pct((prox <= 2).mean())} of phones lie within 2 head-heights of a head.",
                    "The phone–head distance is a useful rule for 'student is using the phone' in Maslak "
                    "(phone near the face/hands vs phone lying on the desk).")
else:
    print("FPI-Det archive not available → section skipped.")
""")

# ---------------------------------------------------------------------------
md(r"""
---
## 6. Near-duplicate audit across splits
The cleaning stage removed **byte-identical** files (MD5). The same scene saved twice, re-encoded, or one second apart has a different MD5, though, and can still sit in train and test at the same time. That kind of leakage inflates test scores. This audit computes a perceptual **difference hash (dHash, 64 bits)** for a sample of images per split. **Stage 1** flags every image that has a hash twin (≤ `NEAR_DUP_BITS` differing bits) in another split. **Stage 2** confirms a candidate only if the two images also match pixel by pixel (correlation ≥ `DUP_MIN_CORR` on 32×32 grayscale). Low-texture images, such as closed eyes, often share a hash without being the same picture, so they are reported as *look-alikes* and not as leakage.
""")

code(r"""
audit_rows, audit_pairs = [], {}

def run_audit(name, df, split_col, n_per_split):
    sample = pd.concat([g.sample(min(n_per_split, len(g)), random_state=SEED) for _, g in df.groupby(split_col)])
    sample = sample.assign(hash=[dhash(p) for p in sample["abs_path"]])
    summ, pairs = cross_split_near_dups(sample, split_col)
    summ.insert(0, "dataset", name)
    audit_rows.append(summ)
    audit_pairs[name] = pairs
    print(f"[{name}] hashed {sample['hash'].notna().sum():,} images across {sample[split_col].nunique()} splits")

def with_paths(df, to_abs, max_rows):
    if to_abs is None:
        return None
    d = df.sample(min(len(df), max_rows), random_state=SEED)
    d = d.assign(abs_path=[str(to_abs(p)) for p in d["path"]])
    d = d[[Path(p).exists() for p in d["abs_path"]]]
    return d if len(d) else None

cap = DUP_SAMPLE_PER_SPLIT * 8
jobs = [
    ("DAiSEE",   with_paths(dai.drop_duplicates("clip"), dai_abs, cap), "split", DUP_SAMPLE_PER_SPLIT),
    ("UTA-RLDD", with_paths(rl, rl_abs, cap), "fold_num", DUP_SAMPLE_PER_SPLIT // 2),
    ("MRL Eye",  with_paths(mrl, mrl_abs, cap), "split", DUP_SAMPLE_PER_SPLIT),
    ("FPI-Det",  fpi_images.rename(columns={"path": "abs_path"}) if fpi_images is not None else None, "split", DUP_SAMPLE_PER_SPLIT * 2),
]
for name, d, split_col, n in jobs:
    if d is None:
        print(f"[{name}] images not available → skipped")
        continue
    run_audit(name, d, split_col, n)

if audit_rows:
    audit = pd.concat(audit_rows, ignore_index=True)
    display(audit)
    audit.to_csv(OUT_DIR / "near_duplicate_audit.csv", index=False)
    per_ds = audit.groupby("dataset").agg(checked=("images", "sum"), candidates=("hash_candidates", "sum"),
                                          confirmed=("confirmed_duplicates", "sum"), max_corr=("max_candidate_corr", "max"))
    display(per_ds)
    for ds, r in per_ds.iterrows():
        if r["confirmed"] == 0:
            add_finding(f"dup_{ds}", ds,
                        f"No confirmed duplicates across splits among {int(r['checked']):,} audited images "
                        f"({int(r['candidates'])} hash look-alikes, highest pixel correlation "
                        f"{0 if pd.isna(r['max_corr']) else r['max_corr']:.2f} < {DUP_MIN_CORR}).",
                        "The split is clean at the visual level too; keep it.")
        else:
            add_finding(f"dup_{ds}", ds,
                        f"{int(r['confirmed']):,} of {int(r['checked']):,} audited images ({pct(r['confirmed'] / r['checked'])}) "
                        "have a confirmed near-identical copy in another split.",
                        "Leakage: group duplicates (same scene / video) and keep each group inside one split, "
                        "or remove the copies from val/test before reporting results.")
    for ds, pairs in audit_pairs.items():
        if ds == "UTA-RLDD":
            continue   # never display frames of subjects who did not allow publication
        if len(pairs):
            kind = "confirmed duplicates" if pairs["confirmed"].any() else "closest look-alikes (NOT duplicates)"
            show_pairs(pairs[pairs["confirmed"]] if pairs["confirmed"].any() else pairs.head(2),
                       f"{ds}: {kind}", "dup_pairs_" + re.sub(r"[^a-z]", "", ds.lower()))
else:
    print("No image archive available → near-duplicate audit skipped.")
""")

md(r"""
---
## 7. Cross-dataset summary
""")


code(r"""
overview = pd.DataFrame([
    dict(dataset="Attention V3", maslak_signal="attention (gaze, head pose, phone, hands)", unit="feature row",
         samples=len(att), subjects="n/a (no id)", classes="attentive / inattentive",
         imbalance=round(imbalance_ratio(att["label"].value_counts()), 2), split="row order 70/15/15",
         id_leak_between_splits="cannot check (no id)"),
    dict(dataset="DAiSEE", maslak_signal="engagement / boredom / confusion / frustration", unit="frame (10 per clip)",
         samples=len(dai), subjects=dai["person"].nunique(), classes="4 states × levels 0-3",
         imbalance=round(share["max/min_ratio"].max(), 1), split="official, person-disjoint",
         id_leak_between_splits=split_overlap(dai, "split", "person")),
    dict(dataset="UTA-RLDD", maslak_signal="drowsiness", unit="frame (1 fps)",
         samples=len(rl), subjects=rl["subject"].nunique(), classes="alert / low_vigilant / drowsy",
         imbalance=round(imbalance_ratio(rl["class_name"].value_counts()), 2), split="5 folds, subject-disjoint",
         id_leak_between_splits=split_overlap(rl, "fold_num", "subject")),
    dict(dataset="MRL Eye", maslak_signal="eye closure", unit="eye crop",
         samples=len(mrl), subjects=mrl["subject"].nunique(), classes="open / closed",
         imbalance=round(imbalance_ratio(mrl["eye_open"].value_counts()), 2), split="subject-disjoint 70/15/15",
         id_leak_between_splits=split_overlap(mrl, "split", "subject")),
    dict(dataset="FPI-Det", maslak_signal="phone use", unit="image + boxes",
         samples=len(fpi_images) if fpi_images is not None else "archive not loaded", subjects="n/a",
         classes="head / phone boxes",
         imbalance=round(imbalance_ratio(fpi_boxes["class"].value_counts()), 1) if fpi_boxes is not None and len(fpi_boxes) else "-",
         split="train / val / test (cleaned)", id_leak_between_splits="duplicates removed in cleaning"),
])
display(overview)
overview.to_csv(OUT_DIR / "dataset_overview.csv", index=False)

add_finding("domain_gap", "All",
            "No dataset was recorded in a study room like Maslak's (online-class webcams, home videos, NIR eye sensors, surveillance views).",
            "Collect a small labelled set of our own study-room recordings and report Accuracy, Precision, Recall and F1 on it "
            "before claiming any final accuracy.")

findings = pd.DataFrame(FINDINGS.values())
findings.index = np.arange(1, len(findings) + 1)
with pd.option_context("display.max_colwidth", None):
    display(findings[["dataset", "finding", "decision"]])
findings.to_csv(OUT_DIR / "eda_findings.csv", index_label="#")
print(f"\n{len(findings)} findings saved to {OUT_DIR / 'eda_findings.csv'}")
""")

md(r"""
### Outputs
Every figure is saved to `eda_outputs/figures/`. The findings table is saved to `eda_outputs/eda_findings.csv` and the overview to `eda_outputs/dataset_overview.csv`. The next cell zips them all so you can download them from the Colab file panel.
""")

code(r"""
archive = shutil.make_archive(str(WORK_DIR / "maslak_eda_outputs"), "zip", OUT_DIR)
print("figures:", len(list(FIG_DIR.glob("*.png"))))
print("download:", archive)
""")

md(r"""
---
### Limitations and next steps
* **Behavioural signals ≠ concentration.** Each dataset labels an observable behaviour (gaze, posture, eye closure, a phone in view). None of them measures understanding.
* **Domain gap.** The datasets come from online classes, home recordings, near-infrared sensors and surveillance angles. Maslak needs its own small, consented, labelled recordings to measure real accuracy.
* **Image checks use samples.** The image statistics come from a random sample (`IMG_PER_GROUP` per class, fixed seed). Raising it gives tighter estimates.
* **Next stage (modelling):** apply the decisions in the findings table: re-stratify Attention V3, binarise DAiSEE, use subject-wise CV for RLDD and MRL, and run per-class evaluation for FPI-Det.
""")


# ---------------------------------------------------------------------------
def build(path):
    nb = {
        "cells": [],
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    for kind, src in CELLS:
        lines = src.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    build(Path(__file__).with_name("Maslak_EDA.ipynb"))
