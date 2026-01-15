import json
from pathlib import Path
from eval_lib.chunking import BalancedSectionChunker, FixedTokensChunker

def test_chunker():
    # Load dataset
    dataset_path = Path(__file__).parent / "datasets" / "qasper-generated" / "data.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Chunk documents
    chunker = BalancedSectionChunker(chunk_size=512, overlap=50)
    chunker.prepare_documents(dataset)
    
    # Prepare output
    output = {}
    for doc_id in chunker.docs:
        output[doc_id] = {
            'title': chunker.docs[doc_id]['title'],
            'chunks': chunker.docs[doc_id]['chunks']
        }
    
    # Save to file
    output_path = Path(__file__).parent / "chunker_test_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(output)} documents to {output_path}")


if __name__ == "__main__":
    test_chunker()
