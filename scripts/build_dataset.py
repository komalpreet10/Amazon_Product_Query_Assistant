from pathlib import Path
import pandas as pd
from datasets import load_dataset

CATEGORY = "raw_review_Electronics"
META_CATEGORY = "raw_meta_Electronics"

TAKE_SIZE = 100_000
SAMPLE_SIZE = 5_000
OUTPUT_PATH = "data/raw/sample.parquet"

def main():
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    print("Loading reviews...")
    reviews = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    CATEGORY,
    split="full",
    streaming=True,
    trust_remote_code=True,
)

    print("Loading metadata...")
    meta = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    META_CATEGORY,
    split="full",
    streaming=True,
    trust_remote_code=True,
)

    reviews_df = pd.DataFrame(list(reviews.take(TAKE_SIZE)))
    meta_df = pd.DataFrame(list(meta.take(TAKE_SIZE)))

    print("Reviews:", reviews_df.shape)
    print("Meta:", meta_df.shape)

    meta_df = meta_df.rename(columns={"title": "product_title"})

    df = reviews_df.merge(meta_df, on="parent_asin", how="inner")
    print("Merged:", df.shape)

    df = df[
        df["text"].notna()
        & df["product_title"].notna()
        & df["average_rating"].notna()
    ].copy()

    keep_cols = [
        "text", "rating", "verified_purchase", "helpful_vote",
        "parent_asin", "user_id", "timestamp",
        "product_title", "price", "average_rating",
        "rating_number", "features", "description",
        "main_category", "store", "categories"
    ]

    df = df[[col for col in keep_cols if col in df.columns]]

    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
    df.to_parquet(OUTPUT_PATH, index=False)

    print("Final:", df.shape)
    print("Saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()