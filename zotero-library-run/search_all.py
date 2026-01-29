import json
import re
from pathlib import Path

import requests

from search import run_search, load_embeddings, EMBEDDINGS_FILE

ZOTERO_API_BASE = "http://localhost:23119/api/groups/6384550"
OUTPUT_FILE = Path(__file__).parent / ".search_all_results.json"


def fetch_all_items() -> list[dict]:
    """Fetch all items from Zotero local API."""
    response = requests.get(f"{ZOTERO_API_BASE}/items")
    response.raise_for_status()
    return response.json()


def get_embedding_notes() -> list[dict]:
    """
    Fetch notes with 'embedding' tag.

    Returns list of dicts with 'question' and 'expected_answer'.
    """
    items = fetch_all_items()
    notes = []

    for item in items:
        data = item.get('data', {})

        # Check if itemType is "note"
        if data.get('itemType') != 'note':
            continue

        # Check if has "embeddings" tag
        tags = data.get('tags', [])
        has_embedding_tag = any(t.get('tag') == 'embeddings-question' for t in tags)
        if not has_embedding_tag:
            continue

        # Get note content and split by <hr>
        note_html = data.get('note', '')
        if not note_html:
            continue

        # Split by <hr> (handle variations like <hr/>, <hr />)
        parts = re.split(r'<hr\s*/?>', note_html, maxsplit=1)
        if len(parts) < 2:
            print(f"  Warning: Note missing <hr> separator, skipping")
            continue

        # Strip HTML tags for clean text
        question = re.sub(r'<[^>]+>', '', parts[0]).strip()
        expected_answer = re.sub(r'<[^>]+>', '', parts[1]).strip()

        if question and expected_answer:
            notes.append({
                'question': question,
                'expected_answer': expected_answer
            })

    return notes


def main():
    print(f"Zotero API: {ZOTERO_API_BASE}")
    print(f"Loading embeddings from: {EMBEDDINGS_FILE}")

    if not EMBEDDINGS_FILE.exists():
        print("Error: Embeddings file not found. Run make_embeddings.py first.")
        return

    embeddings_data = load_embeddings()
    print(f"Loaded {len(embeddings_data['chunks'])} chunks from {len(embeddings_data['item_metadata'])} documents")

    # Fetch notes with embedding tag
    print("\n" + "="*60)
    print("FETCHING NOTES FROM ZOTERO API")
    print("="*60)

    notes = get_embedding_notes()
    print(f"Found {len(notes)} notes with 'embedding' tag")

    if not notes:
        print("No notes found. Exiting.")
        return

    # Run searches for each note
    print("\n" + "="*60)
    print("RUNNING SEARCHES")
    print("="*60)

    results = []
    correct_count = 0
    wrong_count = 0

    for i, note in enumerate(notes, 1):
        print(f"\n[{i}/{len(notes)}] {note['question'][:60]}...")

        result = run_search(
            question=note['question'],
            embeddings_data=embeddings_data,
            top_k=5,
            expected_answer=note['expected_answer']
        )
        results.append(result)

        # Track validation results
        validation = result.get('validation_result', '')
        if validation.startswith('Correct'):
            correct_count += 1
            print(f"  -> Correct")
        else:
            wrong_count += 1
            print(f"  -> Wrong: {validation[:50]}...")

    # Save results
    output = {
        'total': len(results),
        'correct': correct_count,
        'wrong': wrong_count,
        'accuracy': correct_count / len(results) if results else 0,
        'results': results
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {OUTPUT_FILE}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total questions: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Wrong: {wrong_count}")
    print(f"Accuracy: {output['accuracy']:.1%}")


if __name__ == "__main__":
    main()