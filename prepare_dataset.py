"""
Prepare the BBC News Classification dataset for NewsLens.

Recommended source:
https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category

Download/extract the dataset and place its CSV file in this project's data/ folder
(or project root). Common column names such as text, article, content, and category
are detected automatically. Output is saved as data/news_sample.csv, which app.py
already uses.

Expected source columns:
- category: business, entertainment, politics, sport, tech
- text/article/content: full news article text
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT = DATA_DIR / "news_sample.csv"
TEXT_NAMES = {"text", "article", "content", "body", "news", "description"}
CATEGORY_NAMES = {"category", "topic", "label", "class"}

def find_source():
    candidates = list(DATA_DIR.glob("*.csv")) + list(ROOT.glob("*.csv"))
    candidates = [p for p in candidates if p.name.lower() not in {"news_sample.csv"}]
    if not candidates:
        raise FileNotFoundError(
            "BBC dataset CSV not found. Download and extract the BBC News dataset, "
            "then put its CSV file in the data/ folder and run this script again."
        )
    return candidates[0]

def main():
    DATA_DIR.mkdir(exist_ok=True)
    source = find_source()
    print(f"Reading: {source}")
    df = pd.read_csv(source)
    normalized = {str(col).strip().lower(): col for col in df.columns}
    text_col = next((normalized[name] for name in TEXT_NAMES if name in normalized), None)
    category_col = next((normalized[name] for name in CATEGORY_NAMES if name in normalized), None)

    # Some BBC CSV variants use "Article" and "Category"; names are matched case-insensitively.
    if text_col is None or category_col is None:
        raise ValueError(
            f"Could not identify article text/category columns. Found columns: {list(df.columns)}. "
            "Expected a text-like column (text/article/content/body) and category/topic/label."
        )

    out = df[[text_col, category_col]].copy()
    out.columns = ["title", "category"]  # app uses title as the generic text field
    out = out.dropna(subset=["title", "category"])
    out["title"] = out["title"].astype(str).str.strip()
    out["category"] = out["category"].astype(str).str.strip().str.lower()
    out = out[(out["title"] != "") & (out["category"] != "")]
    out = out.drop_duplicates(subset=["title"]).reset_index(drop=True)
    if out["category"].nunique() < 2:
        raise ValueError("Dataset must contain at least two distinct categories.")
    out.to_csv(OUT, index=False)
    print(f"Prepared {len(out):,} articles across {out['category'].nunique()} categories.")
    print(out["category"].value_counts().to_string())
    print(f"Saved prepared dataset to: {OUT}")

if __name__ == "__main__":
    main()
