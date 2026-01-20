import pandas as pd
import os
from pathlib import Path

# Base paths
RAW_DIR = Path(__file__).parent / "raw" / "parquet"
TXT_DIR = Path(__file__).parent / "txt" / "selected"

# Number of articles to extract per language
NUM_ARTICLES = 5000

# Minimum word count for an article to be included
MIN_WORD_COUNT = 1000

# Map parquet filenames to language folder names
PARQUET_FILES = {
    # "spanish.parquet": "spanish",
    # "russian.parquet": "russian",
    "english.parquet": "english",
}


def extract_articles():
    for parquet_file, language in PARQUET_FILES.items():
        parquet_path = RAW_DIR / parquet_file

        if not parquet_path.exists():
            print(f"Warning: {parquet_path} not found, skipping")
            continue

        print(f"Processing {parquet_file}...")

        # Read parquet file
        df = pd.read_parquet(parquet_path)
        print(f"  Total articles: {len(df)}")

        # Create output directory
        output_dir = TXT_DIR / language
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract articles with at least MIN_WORD_COUNT words
        extracted_count = 0

        for idx in range(len(df)):
            if extracted_count >= NUM_ARTICLES:
                break

            row = df.iloc[idx]
            article_id = row["id"]
            title = row["title"]
            text = row["text"]

            # Combine title and text for the full content
            full_text = f"{title}\n\n{text}"

            # Check word count (simple whitespace split)
            word_count = len(full_text.split())
            if word_count < MIN_WORD_COUNT:
                continue

            # Create filename from article ID
            filename = f"{article_id}.txt"
            filepath = output_dir / filename

            # Write article content
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(full_text)

            extracted_count += 1

        print(f"  Extracted {extracted_count} articles to {output_dir}")


if __name__ == "__main__":
    extract_articles()
    print("\nDone!")
