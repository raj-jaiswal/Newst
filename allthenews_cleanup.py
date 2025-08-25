import csv
import sys
from pathlib import Path
import pandas as pd

# raise csv field size limit (handles OverflowError on Windows)
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

SRC_DIR = Path("data") / "allthenews"
OUT_PATH = Path("data") / "NewsContents.csv"
SRC_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(SRC_DIR.glob("*.csv"))
if not files:
    raise FileNotFoundError(f"No CSV files in {SRC_DIR}")

def pick_cols(cols):
    title = next((c for c in cols if "title" in c.lower()), None)
    content = next((c for c in cols if any(k in c.lower() for k in ("content", "article", "text"))), None)
    if title is None and len(cols) >= 3:
        title = cols[2]
    if content is None:
        content = cols[-1]
    return title, content

first_write = True
for f in files:
    print("Processing", f.name)
    for chunk in pd.read_csv(f, dtype=str, encoding="utf-8", engine="python", chunksize=5000):
        cols = list(chunk.columns)
        tcol, ccol = pick_cols(cols)
        sub = chunk[[tcol, ccol]].copy()
        sub.columns = ["title", "content"]
        sub = sub.fillna("")  # avoid NaNs
        sub["title"] = sub["title"].str.replace(r"\s+", " ", regex=True).str.strip()
        sub["content"] = sub["content"].str.replace(r"\s+", " ", regex=True).str.strip()
        sub = sub[(sub["title"] != "") & (sub["content"] != "")]
        # append to CSV (write header only on first chunk)
        sub.to_csv(OUT_PATH, index=False, header=first_write, mode="a", encoding="utf-8")
        first_write = False

print("Done. Combined file written to:", OUT_PATH)
