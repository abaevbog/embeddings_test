"""
Parse SciFact BEIR dataset into a structured JSON format.

Input:
- corpus.jsonl: Documents with _id, title, text (abstract)
- queries.jsonl: Queries with _id, text, metadata (relevant doc IDs)

Output:
- scifact_dataset.json: Combined structured dataset with corpus, queries, and relevance judgments
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import nltk

# Download punkt tokenizer if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def parse_scifact_dataset() -> List[Dict[str, Any]]:
    data_path = Path(__file__).parent
    # data_path = Path(data_dir)
    
    # Load corpus (documents)
    corpus_file = data_path / "raw" / "corpus.jsonl"
    corpus_data = load_jsonl(str(corpus_file))
    
    # Load queries
    queries_file = data_path / "raw" / "queries.jsonl"
    queries_data = load_jsonl(str(queries_file))
    
    # Build document-centric structure
    # First, create a mapping of doc_id -> list of queries
    doc_queries = {}  # doc_id -> list of {query_id, query_text, sentences}
    
    for query in queries_data:
        query_id = query["id"]
        query_text = query["claim"]
        evidence = query.get("evidence", {})
        cited_doc_ids = query.get("cited_doc_ids", [])
        
        # If there's specific evidence (sentence-level annotations)
        if evidence:
            for doc_id, annotations in evidence.items():
                if doc_id not in doc_queries:
                    doc_queries[doc_id] = []
                
                # Collect all sentence indices from all annotations for this query
                all_sentence_indices = []
                for annotation in annotations:
                    sentence_indices = annotation.get("sentences", [])
                    all_sentence_indices.extend(sentence_indices)
                
                doc_queries[doc_id].append({
                    "query_id": query_id,
                    "query_text": query_text,
                    "sentence_indices": all_sentence_indices
                })
        
        # If no evidence but has cited documents, add query with no specific matches
        elif cited_doc_ids:
            for doc_id in cited_doc_ids:
                doc_id_str = str(doc_id)
                if doc_id_str not in doc_queries:
                    doc_queries[doc_id_str] = []
                
                doc_queries[doc_id_str].append({
                    "query_id": query_id,
                    "query_text": query_text,
                    "sentence_indices": []  # No specific sentences
                })
    
    # Build final dataset structure - only include documents with queries
    dataset = []
    for doc in corpus_data:
        doc_id = doc["_id"]
        # Only include documents that have at least one query
        if doc_id in doc_queries:
            # Split document text into sentences
            doc_text = doc.get("text", "")
            sentences = nltk.sent_tokenize(doc_text)
            
            # Convert sentence indices to sentence text
            queries_with_matches = []
            for query_data in doc_queries[doc_id]:
                matches = []
                for idx in query_data["sentence_indices"]:
                    if 0 <= idx < len(sentences):
                        matches.append({"text": sentences[idx]})
                
                queries_with_matches.append({
                    "query_id": query_data["query_id"],
                    "query_text": query_data["query_text"],
                    "matches": matches
                })
            
            doc_entry = {
                "doc_id": doc_id,
                "title": doc.get("title", ""),
                "abstract": doc_text,
                "queries": queries_with_matches
            }
            dataset.append(doc_entry)
    
    return dataset


def main():
    # Define paths
    output_file = Path(__file__).parent / "data.json"
    
    # print(f"Loading SciFact dataset from: {data_dir}")
    
    # Parse dataset
    dataset = parse_scifact_dataset()
    
    # Calculate statistics
    total_docs = len(dataset)
    docs_with_queries = sum(1 for doc in dataset if doc['queries'])
    total_query_annotations = sum(len(doc['queries']) for doc in dataset)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total documents: {total_docs}")
    print(f"  Documents with queries: {docs_with_queries}")
    print(f"  Total query annotations: {total_query_annotations}")
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved dataset to: {output_file}")
    
    # Show example document with queries
    print("\nExample document with queries:")
    example_doc = next((doc for doc in dataset if doc['queries']), None)
    if example_doc:
        print(f"  Doc ID: {example_doc['doc_id']}")
        print(f"  Title: {example_doc['title'][:80]}...")
        print(f"  Abstract: {example_doc['abstract'][:150]}...")
        print(f"  Number of queries: {len(example_doc['queries'])}")
        if example_doc['queries']:
            first_query = example_doc['queries'][0]
            print(f"\n  First query:")
            print(f"    Query ID: {first_query['query_id']}")
            print(f"    Query text: {first_query['query_text'][:80]}...")
            print(f"    Relevant matches ({len(first_query['matches'])}):")
            for i, match in enumerate(first_query['matches'][:2]):  # Show first 2
                print(f"      [{i}]: {match['text'][:80]}...")



if __name__ == "__main__":
    main()
