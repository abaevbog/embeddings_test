import json
import pickle
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import boto3
import requests

runtime = boto3.client('sagemaker-runtime')

CHUNK_SIZE_WORDS = 550  # Rough approximation of 768 tokens
CHUNK_OVERLAP_WORDS = 40

ZOTERO_API_BASE = "http://localhost:23119/api/groups/6384550"
SAGEMAKER_ENDPOINT = "Snowflake-snowflake-arctic-embed-l-v2-0-endpoint"
OUTPUT_FILE = Path(__file__).parent / ".embeddings"


def fetch_all_items() -> list[dict]:
    """Fetch all items from Zotero local API."""
    response = requests.get(f"{ZOTERO_API_BASE}/items")
    response.raise_for_status()
    return response.json()


def fetch_fulltext(item_key: str) -> str | None:
    """Fetch fulltext for an item from Zotero local API."""
    try:
        response = requests.get(f"{ZOTERO_API_BASE}/items/{item_key}/fulltext")
        response.raise_for_status()
        data = response.json()
        return data.get('content', '')
    except requests.exceptions.HTTPError:
        return None


def fetch_item_title(item_key: str) -> str:
    """Fetch title for an item from Zotero local API."""
    try:
        response = requests.get(f"{ZOTERO_API_BASE}/items/{item_key}")
        response.raise_for_status()
        data = response.json()
        return data.get('data', {}).get('title', '')
    except requests.exceptions.HTTPError:
        return ''


def get_pdf_items_with_metadata() -> list[dict]:
    """
    Fetch all PDF items and resolve their titles from parent items.

    Returns list of dicts with 'item_key' and 'title'.
    """
    items = fetch_all_items()
    pdf_items = []

    for item in items:
        data = item.get('data', {})
        content_type = data.get('contentType', '')

        if content_type == 'application/pdf':
            item_key = data.get('key', '')
            parent_key = data.get('parentItem', '')

            # Get title from parent item if it exists, otherwise use item's own title
            if parent_key:
                title = fetch_item_title(parent_key)
            else:
                title = data.get('title', '')

            pdf_items.append({
                'item_key': item_key,
                'title': title
            })

    return pdf_items


def chunk_text(text: str, title: str = "") -> list[str]:
    if not text:
        return []

    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS)

    for i in range(0, len(words), step):
        chunk_words = words[i:i + CHUNK_SIZE_WORDS]
        if chunk_words:
            chunk = ' '.join(chunk_words)
            # Prepend title to each chunk
            if title:
                chunk = f"Title: {title}\n\n{chunk}"
            chunks.append(chunk)

    return chunks


def embed_batch(texts: list[str], doc_prefix: str = "") -> np.ndarray:
    if not texts:
        return np.array([])

    prefixed = [f"{doc_prefix}{t}" for t in texts] if doc_prefix else texts

    # TEI has a batch size limit of 32
    MAX_BATCH_SIZE = 32
    MAX_WORKERS = 10

    def embed_single_batch(batch: list[str]) -> np.ndarray:
        response = runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType='application/json',
            Body=json.dumps({"inputs": batch})
        )
        result = json.loads(response['Body'].read().decode())
        return np.array(result, dtype=np.float32)

    # Split into batches
    batches = [prefixed[i:i + MAX_BATCH_SIZE] for i in range(0, len(prefixed), MAX_BATCH_SIZE)]

    # Process batches in parallel
    all_embeddings = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(embed_single_batch, batch): idx for idx, batch in enumerate(batches)}
        for future in futures:
            idx = futures[future]
            all_embeddings[idx] = future.result()

    return np.vstack(all_embeddings) if all_embeddings else np.array([])


def save_embeddings(embeddings_data: dict, output_path: Path):
    """
    Save embeddings data to file.

    Format:
    {
        'embeddings': np.ndarray,  # Shape: (num_chunks, embedding_dim)
        'chunk_to_doc_map': [(item_key, chunk_idx), ...],  # Maps each embedding to its source
        'chunks': [str, ...],  # Original chunk texts for reference
        'item_metadata': {item_key: {'title': str, 'num_chunks': int, 'text_length': int}, ...},
        'endpoint_name': str
    }
    """
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings_data, f)
    print(f"\nSaved embeddings to: {output_path}")


def main():
    print(f"Zotero API: {ZOTERO_API_BASE}")
    print(f"SageMaker endpoint: {SAGEMAKER_ENDPOINT}")
    print(f"Chunk size: {CHUNK_SIZE_WORDS} words (~{int(CHUNK_SIZE_WORDS * 1.3)} tokens)")

    # Fetch PDF items from Zotero API
    print("\n" + "="*60)
    print("FETCHING ITEMS FROM ZOTERO API")
    print("="*60)

    pdf_items = get_pdf_items_with_metadata()
    print(f"Found {len(pdf_items)} PDF items")

    if not pdf_items:
        print("Error: No PDF items found.")
        return

    # Process each item
    all_chunks = []
    chunk_to_doc_map = []
    item_metadata = {}

    print("\n" + "="*60)
    print("PROCESSING DOCUMENTS")
    print("="*60)

    for i, item in enumerate(pdf_items, 1):
        item_key = item['item_key']
        title = item['title']
        print(f"\n[{i}/{len(pdf_items)}] Processing: {item_key}")
        print(f"  Title: {title}")

        # Fetch fulltext from API
        text = fetch_fulltext(item_key)
        if not text:
            print(f"  Skipping - no fulltext available")
            continue

        print(f"  Text length: {len(text):,} characters")

        # Chunk the text (with title prepended to each chunk)
        chunks = chunk_text(text, title)
        print(f"  Chunks: {len(chunks)}")

        if not chunks:
            print(f"  Skipping - no chunks generated")
            continue

        # Track chunk mappings
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_to_doc_map.append((item_key, chunk_idx))

        # Store metadata
        item_metadata[item_key] = {
            'title': title,
            'num_chunks': len(chunks),
            'text_length': len(text)
        }

    if not all_chunks:
        print("\nNo chunks to process. Exiting.")
        return

    print("\n" + "="*60)
    print("GENERATING EMBEDDINGS")
    print("="*60)
    print(f"Total chunks to embed: {len(all_chunks)}")

    # Generate embeddings for all chunks
    embeddings = embed_batch(all_chunks)
    print(f"Embeddings shape: {embeddings.shape}")

    # Prepare output data
    embeddings_data = {
        'embeddings': embeddings,
        'chunk_to_doc_map': chunk_to_doc_map,
        'chunks': all_chunks,
        'item_metadata': item_metadata,
        'endpoint_name': SAGEMAKER_ENDPOINT
    }

    # Save to file
    save_embeddings(embeddings_data, OUTPUT_FILE)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Items processed: {len(item_metadata)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Embedding dimensions: {embeddings.shape[1] if len(embeddings.shape) > 1 else 'N/A'}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nPer-item breakdown:")
    for item_key, meta in item_metadata.items():
        print(f"  {item_key}: {meta['num_chunks']} chunks, {meta['text_length']:,} chars - {meta['title']}")


if __name__ == "__main__":
    main()
