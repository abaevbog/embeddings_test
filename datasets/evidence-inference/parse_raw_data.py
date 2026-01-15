import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any


def load_prompts(file_path: str) -> pd.DataFrame:
    """Load prompts CSV and construct query text."""
    df = pd.read_csv(file_path)
    
    # Construct query text from Outcome/Intervention/Comparator
    df['query_text'] = df.apply(
        lambda row: f"With respect to {row['Outcome']}, characterize the reported difference between {row['Intervention']} and those receiving {row['Comparator']}.",
        axis=1
    )
    
    return df


def load_annotations(file_path: str) -> pd.DataFrame:
    """Load annotations CSV with evidence."""
    return pd.read_csv(file_path)


def load_full_text(txt_dir: Path, pmcid: int) -> str:
    """Load full text from txt file."""
    txt_file = txt_dir / f"PMC{pmcid}.txt"
    if txt_file.exists():
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Warning: Could not read {txt_file}: {e}")
            return ""
    return ""


def extract_evidence_from_text(full_text: str, start: int, end: int) -> str:
    """Extract evidence text using start/end positions."""
    if start >= 0 and end > start and len(full_text) > end:
        return full_text[start:end]
    return ""


def deduplicate_matches(matches: List[Dict[str, str]], max_word_diff: int = 5) -> List[Dict[str, str]]:
    """
    Remove near-duplicate matches that differ by ≤ max_word_diff words.
    
    Strategy:
    - Compare each pair of matches by word count difference
    - If difference ≤ max_word_diff, keep the longer one (more complete)
    """
    if not matches:
        return []
    
    # Sort by length (longest first) to prefer keeping longer versions
    sorted_matches = sorted(matches, key=lambda m: len(m['text']), reverse=True)
    
    unique_matches = []
    
    for match in sorted_matches:
        match_words = match['text'].split()
        is_duplicate = False
        
        # Check against already accepted unique matches
        for unique_match in unique_matches:
            unique_words = unique_match['text'].split()
            word_diff = abs(len(match_words) - len(unique_words))
            
            # If word count difference is small, check if they're similar
            if word_diff <= max_word_diff:
                # Check if one is a substring of the other (case-insensitive)
                match_lower = match['text'].lower()
                unique_lower = unique_match['text'].lower()
                
                if match_lower in unique_lower or unique_lower in match_lower:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_matches.append(match)
    
    return unique_matches


def parse_evidence_inference_dataset() -> List[Dict[str, Any]]:
    data_path = Path(__file__).parent
    
    # Load data
    prompts_df = load_prompts(str(data_path / "raw" / "prompts_merged.csv"))
    annotations_df = load_annotations(str(data_path / "raw" / "annotations_merged.csv"))
    txt_dir = data_path / "raw" / "txt_files"
    
    # Group by PMCID to process documents
    dataset = []
    processed_docs = set()
    
    for pmcid in prompts_df['PMCID'].unique():
        # Skip if already processed
        if pmcid in processed_docs:
            continue
        processed_docs.add(pmcid)
        
        # Load full text
        full_text = load_full_text(txt_dir, pmcid)
        if not full_text:
            print(f"Warning: No full text found for PMCID {pmcid}, skipping...")
            continue
        
        # Get all prompts for this document
        doc_prompts = prompts_df[prompts_df['PMCID'] == pmcid]
        
        # Build queries with evidence
        queries = []
        for _, prompt_row in doc_prompts.iterrows():
            prompt_id = prompt_row['PromptID']
            query_text = prompt_row['query_text']
            
            # Get all annotations for this prompt
            prompt_annotations = annotations_df[
                (annotations_df['PromptID'] == prompt_id) & 
                (annotations_df['PMCID'] == pmcid)
            ]
            
            # Skip if no valid annotations
            if len(prompt_annotations) == 0:
                continue
            
            # Collect evidence texts
            evidence_texts = set()
            
            for _, ann_row in prompt_annotations.iterrows():
                # Skip invalid annotations
                if not ann_row.get('Valid Label', False):
                    continue
                
                # Try to extract evidence from position
                start = ann_row.get('Evidence Start', -1)
                end = ann_row.get('Evidence End', -1)
                
                if start >= 0 and end > start:
                    evidence = extract_evidence_from_text(full_text, start, end)
                    if evidence:
                        evidence_texts.add(evidence.strip())
                
                # Also use the Annotations field if available
                annotations_text = ann_row.get('Annotations', '')
                if isinstance(annotations_text, str) and annotations_text.strip():
                    # Skip if it looks like HTML or table markup
                    if not annotations_text.startswith('<') and len(annotations_text) > 10:
                        evidence_texts.add(annotations_text.strip())
            
            # Skip queries with no evidence
            if not evidence_texts:
                continue
            
            # Convert to matches format
            matches = [{"text": text} for text in evidence_texts]
            
            # Deduplicate near-duplicate matches
            matches = deduplicate_matches(matches, max_word_diff=5)
            
            # Skip if deduplication removed all matches
            if not matches:
                continue
            
            queries.append({
                "query_id": str(prompt_id),
                "query_text": query_text,
                "matches": matches
            })
        
        # Skip documents with no valid queries
        if not queries:
            continue
        
        # Create document entry
        # Note: Evidence-Inference doesn't have structured sections or abstracts
        # The txt files contain the full text in plain format
        doc_entry = {
            "doc_id": str(pmcid),
            "title": f"PMC{pmcid}",  # No title available, use PMCID
            "abstract": "",  # No structured abstract in txt files
            "full_text": [
                {
                    "section_name": "Full Text",
                    "paragraphs": [full_text]
                }
            ],
            "queries": queries
        }
        dataset.append(doc_entry)
    
    return dataset


def main():
    output_file = Path(__file__).parent / "data.json"
    
    print("Parsing Evidence-Inference dataset...")
    dataset = parse_evidence_inference_dataset()
    
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved dataset to: {output_file}")


if __name__ == "__main__":
    main()
