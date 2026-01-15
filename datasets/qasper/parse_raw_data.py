"""
Parse QASPER dataset into a structured JSON format.

Input:
- qasper-train-v0.3.json: Documents with QA pairs

Output:
- data.json: Combined structured dataset with documents and queries
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def load_qasper_dataset(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_qasper_dataset() -> List[Dict[str, Any]]:
    data_path = Path(__file__).parent
    
    # Load QASPER data
    qasper_file = data_path / "raw" / "qasper-train-v0.3.json"
    qasper_data = load_qasper_dataset(str(qasper_file))
    
    dataset = []
    
    for doc_id, doc_data in qasper_data.items():
        # Extract basic document fields
        title = doc_data.get("title", "")
        abstract = doc_data.get("abstract", "")
        full_text = doc_data.get("full_text", [])
        
        # Process queries from qas
        queries = []
        for qa in doc_data.get("qas", []):
            query_id = qa.get("question_id", "")
            query_text = qa.get("question", "")
            
            # Aggregate evidence from all answers
            evidence_texts = set()  # Use set to avoid duplicates
            for answer_data in qa.get("answers", []):
                answer = answer_data.get("answer", {})
                
                # Skip unanswerable questions
                if answer.get("unanswerable", False):
                    continue
                
                evidence_list = answer.get("evidence", [])
                
                # Add all evidence texts
                for evidence in evidence_list:
                    if evidence:  # Skip empty strings
                        evidence_texts.add(evidence)
            
            # Skip questions with no evidence
            if not evidence_texts:
                continue
            
            # Convert evidence to matches format
            matches = [{"text": text} for text in evidence_texts]
            
            queries.append({
                "query_id": query_id,
                "query_text": query_text,
                "matches": matches
            })
        
        figures = []
        for fig in doc_data.get("figures_and_tables", []):
            caption = fig.get("caption", "")
            if caption:
                figures.append(caption)
        
        if len(figures) > 0:
            full_text.append({ "section_name": "Figures", "paragraphs": ["\n\n".join(figures)] })
        # Only include documents that have queries
        if queries:
            doc_entry = {
                "doc_id": doc_id,
                "title": title,
                "abstract": abstract,
                "full_text": full_text,
                "queries": queries
            }
            dataset.append(doc_entry)
    
    return dataset


def main():
    output_file = Path(__file__).parent / "data.json"
    
    print("Parsing QASPER dataset...")
    dataset = parse_qasper_dataset()
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved dataset to: {output_file}")

if __name__ == "__main__":
    main()
