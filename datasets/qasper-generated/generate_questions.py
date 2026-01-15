"""
Generate questions from QASPER papers using Claude API.

This script reads QASPER papers, combines their content, and uses Claude
to generate 5 questions with evidence passages for each paper.
"""
import json
import os
from pathlib import Path
from anthropic import Anthropic

# Configuration
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    raise ValueError("Please set ANTHROPIC_API_KEY environment variable")

INPUT_FILE = Path(__file__).parent / "raw" / "qasper-train-v0.3.json"
OUTPUT_DIR = Path(__file__).parent / "raw" / "generated_qa"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize Claude client
client = Anthropic(api_key=API_KEY)


def combine_paper_text(paper_data):
    """Combine title, abstract, and full_text into a single string."""
    parts = []
    
    # Add title
    if paper_data.get("title"):
        parts.append(f"Title: {paper_data['title']}\n")
    
    # Add abstract
    if paper_data.get("abstract"):
        parts.append(f"Abstract: {paper_data['abstract']}\n")
    
    # Add full text sections
    if paper_data.get("full_text"):
        for section in paper_data["full_text"]:
            section_name = section.get("section_name", "")
            paragraphs = section.get("paragraphs", [])
            
            if section_name:
                parts.append(f"\n{section_name}:\n")
            
            for paragraph in paragraphs:
                if paragraph:
                    parts.append(f"{paragraph}\n")
    
    return "\n".join(parts)


def generate_questions(paper_text, doc_id):
    """Use Claude API to generate questions with evidence passages."""
    
    prompt = f"""You are analyzing a scientific paper. Read the paper below and generate exactly 5 questions about it.

IMPORTANT: Formulate questions as general, context-free queries that could be asked about ANY paper on this topic.

DO NOT use phrases like:
- "What do the authors..."
- "In this paper..."
- "What does this study..."
- "What approach is proposed..."

INSTEAD, formulate questions generically:
- "What methods exist for [topic]?"
- "How can [problem] be addressed?"
- "What are the challenges in [domain]?"

The questions should be about the paper's topic domain, NOT about "this specific paper" or "the authors".

CRITICAL EVIDENCE REQUIREMENTS:
For each question, the evidence MUST:
1. Actually ANSWER the question with specific details, not just acknowledge the topic exists
2. Be SUFFICIENT to answer the question (or a substantial part of it)
3. Contain concrete information, examples, methods, or explanations
4. NOT be merely a statement that something exists or is important
5. Be an EXACT, VERBATIM quote from the paper - copy-paste the text EXACTLY as it appears
6. Do NOT rephrase, paraphrase, or summarize - use the EXACT words from the paper

BAD EVIDENCE (avoid this):
- Question: "What challenges exist in X?"
- Evidence: "Challenges exist when working with X." ❌ (just acknowledges challenges exist, doesn't say WHAT they are)
- Evidence: "They found that challenge A occurs..." ❌ (rephrased/paraphrased, not exact quote)

GOOD EVIDENCE (do this):
- Question: "What challenges exist in X?"
- Evidence: "Challenge A occurs because of Y. Challenge B happens when Z..." ✓ (actually lists specific challenges AND is exact quote)

For each question:
1. The question should be general and context-free (as if asking about the field, not this specific paper)
2. Provide the exact passage(s) from the paper that ACTUALLY ANSWERS the question with specific details
3. The evidence MUST be an EXACT, VERBATIM quote - copy the text exactly as written in the paper
4. Do NOT paraphrase or rephrase the evidence - it must be word-for-word from the paper
5. If you cannot find evidence that truly answers the question, skip that question and create a different one

Format your response exactly like this for each question:

QUESTION: [your general, context-free question here]
EVIDENCE: [exact quote from paper that contains the actual answer]
EVIDENCE: [another exact quote if needed to complete the answer]

[blank line between questions]

Paper:
{paper_text}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = message.content[0].text
        
        # Parse the response
        questions = []
        current_question = None
        current_evidence = []
        
        for line in response_text.split('\n'):
            line = line.strip()
            
            if line.startswith("QUESTION:"):
                # Save previous question if exists
                if current_question:
                    questions.append({
                        "question": current_question,
                        "evidence": current_evidence
                    })
                
                # Start new question
                current_question = line[9:].strip()  # Remove "QUESTION:" prefix
                current_evidence = []
            
            elif line.startswith("EVIDENCE:"):
                evidence_text = line[9:].strip()  # Remove "EVIDENCE:" prefix
                if evidence_text:
                    current_evidence.append(evidence_text)
        
        # Don't forget the last question
        if current_question:
            questions.append({
                "question": current_question,
                "evidence": current_evidence
            })
        
        # Validate that all evidence is actually in the paper text
        validated_questions = []
        for q in questions:
            valid_evidence = []
            for evidence in q['evidence']:
                # Check if evidence is a substring of the paper (case-sensitive)
                if evidence in paper_text:
                    valid_evidence.append(evidence)
                else:
                    print(f"  Warning: Evidence not found in paper: '{evidence[:100]}...'")
            
            # Only include question if it has at least one valid evidence
            if valid_evidence:
                validated_questions.append({
                    "question": q['question'],
                    "evidence": valid_evidence
                })
            else:
                print(f"  Warning: Skipping question with no valid evidence: '{q['question']}'")
        
        return {
            "doc_id": doc_id,
            "questions": validated_questions
        }
    
    except Exception as e:
        print(f"Error generating questions for {doc_id}: {e}")
        return None


def main():
    print(f"Reading papers from: {INPUT_FILE}")
    # Read the QASPER data
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    print(f"Found {len(papers)} papers")
    
    # Process each paper
    for idx, (doc_id, paper_data) in enumerate(papers.items(), 1):
        if idx > 101:
            break
        print(f"\n[{idx}/{len(papers)}] Processing paper: {doc_id}")
        
        # Check if already processed
        output_file = OUTPUT_DIR / f"{doc_id}.json"
        if output_file.exists():
            print(f"  Skipping - already exists")
            continue
        
        # Combine paper text
        paper_text = combine_paper_text(paper_data)
        
        # Check if paper has enough content
        if len(paper_text) < 500:
            print(f"  Skipping - insufficient content")
            continue
        
        print(f"  Paper length: {len(paper_text)} characters")
        
        # Generate questions using Claude
        result = generate_questions(paper_text, doc_id)
        
        if result:
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(result['questions'])} questions to {output_file.name}")
        else:
            print(f"  Failed to generate questions")
            break
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
