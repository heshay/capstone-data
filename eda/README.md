# EDA — MASLK Smart Study Room

Exploratory data analysis of the cleaned datasets in this repo. It reads the cleaned CSVs from this repo and does not change them.

| File | What it is |
|---|---|
| `Maslak_EDA.ipynb` | The notebook (open it in Google Colab and run all) |
| `build_notebook.py` | Generates the notebook. Edit this file, then run `python build_notebook.py` |

## How to run

1. Open `Maslak_EDA.ipynb` in Google Colab.
2. For the image checks, paste the OneDrive folder link with the image zips into `ARCHIVE_FOLDER_URL` in section 0.2. Ask Hussah for the link. It is not stored here because the images include faces.
3. Runtime → Run all. Without the link, the table analysis still runs and the image checks are skipped.

The notebook saves every figure and a findings table (`eda_findings.csv`) to `eda_outputs/`.

## Main findings

**Attention V3**
- The inattentive share is different in each split: train 43.1%, val 57.0%, test 62.3% (χ² p = 2.7e-33). This happens because the split was made by row order.
- Rows behave like consecutive video frames, so a random shuffle would leak near-identical rows into test.
- Decision: re-split into contiguous blocks with the same label share in each split.

**DAiSEE**
- No person appears in more than one split, and no duplicate images were found across splits.
- The levels are very unbalanced: engagement level 0 is 0.7% of clips, and the rarest frustration level is 1.0%.
- Consecutive frames of a clip are near-duplicates: pixel difference 1.2, compared with 16.0 between clips.
- Decision: binarise the labels, use class weights, and treat the clip as the unit.

**UTA-RLDD**
- The three states are balanced, and the folds are subject-disjoint.
- Frames 1 s apart are nearly identical: pixel difference 5.5, compared with 11.0 at 60 s.
- Decision: sample one frame every 5–10 s, and show only publishable subjects in reports.

**MRL Eye**
- The split is by subject, and no duplicates were confirmed across splits.

**FPI-Det**
- Class ids from `data.yaml`: 0 = phone, 1 = head. This was confirmed by box size, since phones are smaller.
- Phones are small objects, with a median box of 1.7% of the image. Decision: train at 640 px or more.
- 90.9% of phones lie within 2 head-heights of a head. This is useful for a "using the phone" rule.
- The `val` split is different from the others: 26.2% of its images show a phone without a head box, compared with about 0% in train and test. Head labels may be missing there. Until this is checked, report metrics on `test` and per class.
- 12 of 6,691 audited images (0.2%) have a near-identical copy in another split.

**Across all datasets:** split the data by person, because the same person in both train and test makes accuracy look better than it really is.
