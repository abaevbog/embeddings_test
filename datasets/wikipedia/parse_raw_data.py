#!/usr/bin/env python3
"""
Parse Wikipedia dataset into standard format.
Combines article content from raw/selected/{language}/*.txt with
generated questions from raw/generated_qa/{language}/*.json
"""

import json
from pathlib import Path


def read_article(filepath: Path) -> tuple[str, str]:
    """Read article from file. Returns (title, text)."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # First line is title, rest is text (after blank line)
    lines = content.split("\n", 2)
    title = lines[0].strip()
    text = lines[2].strip() if len(lines) > 2 else ""

    return title, text


def load_generated_questions(generated_qa_dir: Path) -> dict:
    """Load all generated Q&A files from all language subdirectories."""
    print(f"Loading generated questions from {generated_qa_dir}")

    generated_data = {}

    # Iterate through language subdirectories
    for lang_dir in generated_qa_dir.iterdir():
        if not lang_dir.is_dir() or lang_dir.name.startswith('.'):
            continue

        language = lang_dir.name
        generated_data[language] = {}

        qa_files = list(lang_dir.glob("*.json"))
        print(f"  Found {len(qa_files)} Q&A files for {language}")

        for qa_file in qa_files:
            with open(qa_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                doc_id = data["doc_id"]
                generated_data[language][doc_id] = data

    return generated_data


def create_dataset(selected_dir: Path, generated_data: dict) -> list:
    """Create the final dataset in standard format."""
    dataset = []

    for language, qa_data_by_doc in generated_data.items():
        lang_articles_dir = selected_dir / language

        if not lang_articles_dir.exists():
            print(f"Warning: Articles directory {lang_articles_dir} not found, skipping")
            continue

        print(f"Processing {language}: {len(qa_data_by_doc)} documents")

        for doc_id, qa_data in qa_data_by_doc.items():
            article_path = lang_articles_dir / f"{doc_id}.txt"

            if not article_path.exists():
                print(f"  Warning: Article {article_path} not found, skipping")
                continue

            # Read article content
            title, text = read_article(article_path)

            # Create full_text structure (single section with all paragraphs)
            # Split text into paragraphs by double newlines
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            full_text = [
                {
                    "section_name": "",
                    "paragraphs": paragraphs
                }
            ]

            # Convert questions to queries format
            queries = []
            for i, q in enumerate(qa_data["questions"]):
                query = {
                    "query_id": f"{doc_id}_q{i}",
                    "query_text": q["question"],
                    "matches": [{"text": evidence} for evidence in q["evidence"]]
                }
                queries.append(query)

            # Add to dataset
            dataset.append({
                "doc_id": f"{language}_{doc_id}",
                "title": title,
                "abstract": "",
                "full_text": full_text,
                "queries": queries
            })

    print(f"Created dataset with {len(dataset)} documents")
    return dataset


def main():
    # Paths
    script_dir = Path(__file__).parent
    selected_dir = script_dir / "raw" / "selected"
    generated_qa_dir = script_dir / "raw" / "generated_qa"
    output_path = script_dir / "data.json"

    # Load generated questions
    generated_data = load_generated_questions(generated_qa_dir)

    # Create dataset
    dataset = create_dataset(selected_dir, generated_data)

    # Save output
    print(f"Saving dataset to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    # Print statistics
    print("\nDataset statistics:")
    print(f"  Total documents: {len(dataset)}")
    total_queries = sum(len(doc["queries"]) for doc in dataset)
    print(f"  Total queries: {total_queries}")
    total_evidence = sum(
        len(query["matches"])
        for doc in dataset
        for query in doc["queries"]
    )
    print(f"  Total evidence spans: {total_evidence}")
    if dataset:
        print(f"  Average queries per document: {total_queries / len(dataset):.1f}")
    if total_queries:
        print(f"  Average evidence per query: {total_evidence / total_queries:.1f}")


if __name__ == "__main__":
    main()
