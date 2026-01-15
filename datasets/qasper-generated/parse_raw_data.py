#!/usr/bin/env python3
"""
Parse QASPER-generated dataset into standard format.
Combines paper content from qasper-train-v0.3.json with generated questions from generated_qa/*.json
"""

import json
import os
from pathlib import Path


def load_qasper_papers(qasper_path):
    """Load original QASPER papers."""
    print(f"Loading QASPER papers from {qasper_path}")
    with open(qasper_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    print(f"Loaded {len(papers)} papers")
    return papers


def extract_full_text_structure(paper_data):
    """Extract full text sections preserving structure."""
    if 'full_text' not in paper_data:
        return []
    
    sections = []
    for section in paper_data['full_text']:
        sections.append({
            'section_name': section.get('section_name', ''),
            'paragraphs': section.get('paragraphs', [])
        })
    
    return sections


def load_generated_questions(generated_qa_dir):
    """Load all generated Q&A files."""
    print(f"Loading generated questions from {generated_qa_dir}")
    
    qa_files = list(Path(generated_qa_dir).glob('*.json'))
    print(f"Found {len(qa_files)} Q&A files")
    
    generated_data = {}
    for qa_file in qa_files:
        with open(qa_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            doc_id = data['doc_id']
            generated_data[doc_id] = data
    
    return generated_data


def create_dataset(papers, generated_data):
    """Create the final dataset in standard format."""
    dataset = []
    
    for doc_id, qa_data in generated_data.items():
        if doc_id not in papers:
            print(f"Warning: Paper {doc_id} not found in QASPER data, skipping")
            continue
        
        paper = papers[doc_id]
        
        # Extract full text structure
        full_text = extract_full_text_structure(paper)
        
        # Convert questions to queries format
        queries = []
        for i, q in enumerate(qa_data['questions']):
            query = {
                'query_id': f"{doc_id}_q{i}",
                'query_text': q['question'],
                'matches': [{'text': evidence} for evidence in q['evidence']]
            }
            queries.append(query)
        
        # Add to dataset as array element
        dataset.append({
            'doc_id': doc_id,
            'title': paper.get('title', ''),
            'abstract': paper.get('abstract', ''),
            'full_text': full_text,
            'queries': queries
        })
    
    print(f"Created dataset with {len(dataset)} documents")
    return dataset


def main():
    # Paths
    script_dir = Path(__file__).parent
    qasper_path = script_dir / 'raw' / 'qasper-train-v0.3.json'
    generated_qa_dir = script_dir / 'raw' / 'generated_qa'
    output_path = script_dir / 'data.json'
    
    # Load data
    papers = load_qasper_papers(qasper_path)
    generated_data = load_generated_questions(generated_qa_dir)
    
    # Create dataset
    dataset = create_dataset(papers, generated_data)
    
    # Save output
    print(f"Saving dataset to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print("\nDataset statistics:")
    print(f"  Total documents: {len(dataset)}")
    total_queries = sum(len(doc['queries']) for doc in dataset)
    print(f"  Total queries: {total_queries}")
    total_evidence = sum(
        len(query['matches']) 
        for doc in dataset 
        for query in doc['queries']
    )
    print(f"  Total evidence spans: {total_evidence}")
    print(f"  Average queries per document: {total_queries / len(dataset):.1f}")
    print(f"  Average evidence per query: {total_evidence / total_queries:.1f}")


if __name__ == '__main__':
    main()
