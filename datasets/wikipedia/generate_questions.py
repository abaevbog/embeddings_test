"""
Generate questions from Wikipedia articles using Claude API.

This script reads Wikipedia articles from the selected folder and uses Claude
to generate questions with evidence passages for each article.
"""
import json
import os
import time
from pathlib import Path
from anthropic import Anthropic

# Configuration
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    raise ValueError("Please set ANTHROPIC_API_KEY environment variable")

BASE_DIR = Path(__file__).parent
SELECTED_DIR = BASE_DIR / "raw" / "selected"
OUTPUT_DIR = BASE_DIR / "raw" / "generated_qa"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize Claude client
client = Anthropic(api_key=API_KEY)


def read_article(filepath: Path) -> tuple[str, str]:
    """Read article from file. Returns (title, full_text)."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # First line is title, rest is text
    lines = content.split("\n", 2)
    title = lines[0].strip()
    text = lines[2].strip() if len(lines) > 2 else ""

    return title, text


def generate_questions(article_text: str, title: str, doc_id: str, language: str):
    """Use Claude API to generate questions with evidence passages."""

    # Language-specific prompts
    language_instructions = {
        "spanish": "Generate questions and evidence in Spanish (the same language as the article).",
        "russian": "Generate questions and evidence in Russian (the same language as the article).",
        "german": "Generate questions and evidence in German (the same language as the article).",
        "chinese": "Generate questions and evidence in Chinese (the same language as the article).",
        "english": "Generate questions and evidence in English.",
    }

    lang_instruction = language_instructions.get(language, "Generate questions in the same language as the article.")

    prompt = f"""You are analyzing a Wikipedia article. Read the article below and generate exactly 5 questions about it.

{lang_instruction}

IMPORTANT: Formulate questions as general, context-free queries that could be asked about this topic.

DO NOT use phrases like:
- "What does the article say..."
- "According to this article..."
- "What is mentioned..."

INSTEAD, formulate questions generically:
- "What is [topic]?"
- "How did [event] happen?"
- "What are the main characteristics of [subject]?"

The questions should be about the topic domain, NOT about "this specific article".

CRITICAL EVIDENCE REQUIREMENTS:
For each question, the evidence MUST:
1. Actually ANSWER the question with specific details, not just acknowledge the topic exists
2. Be SUFFICIENT to answer the question (or a substantial part of it)
3. Contain concrete information, examples, facts, or explanations
4. NOT be merely a statement that something exists or is important
5. Be an EXACT, VERBATIM quote from the article - copy-paste the text EXACTLY as it appears
6. Do NOT rephrase, paraphrase, or summarize - use the EXACT words from the article

BAD EVIDENCE (avoid this):
- Question: "What challenges exist in X?"
- Evidence: "Challenges exist when working with X." (just acknowledges challenges exist, doesn't say WHAT they are)

GOOD EVIDENCE (do this):
- Question: "What challenges exist in X?"
- Evidence: "Challenge A occurs because of Y. Challenge B happens when Z..." (actually lists specific challenges AND is exact quote)

For each question:
1. The question should be general and context-free (as if asking about the topic, not this specific article)
2. Provide the exact passage(s) from the article that ACTUALLY ANSWERS the question with specific details
3. The evidence MUST be an EXACT, VERBATIM quote - copy the text exactly as written in the article
4. Do NOT paraphrase or rephrase the evidence - it must be word-for-word from the article
5. If you cannot find evidence that truly answers the question, skip that question and create a different one

Format your response exactly like this for each question:

QUESTION: [your general, context-free question here]
EVIDENCE: [exact quote from article that contains the actual answer]
EVIDENCE: [another exact quote if needed to complete the answer]

[blank line between questions]

Article Title: {title}

Article:
{article_text}"""

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

        # Validate that all evidence is actually in the article text
        validated_questions = []
        for q in questions:
            valid_evidence = []
            for evidence in q['evidence']:
                # Check if evidence is a substring of the article (case-sensitive)
                if evidence in article_text:
                    valid_evidence.append(evidence)
                else:
                    print(f"  Warning: Evidence not found in article: '{evidence[:100]}...'")

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
            "language": language,
            "title": title,
            "questions": validated_questions
        }

    except Exception as e:
        print(f"Error generating questions for {doc_id}: {e}")
        return None


def main():
    # Get all language folders
    languages = ['english']
    print(f"Found languages: {languages}")

    total_processed = 0

    for language in languages:
        lang_dir = SELECTED_DIR / language
        output_lang_dir = OUTPUT_DIR / language
        output_lang_dir.mkdir(exist_ok=True)

        # Get all article files
        article_files = list(lang_dir.glob("*.txt"))
        print(f"\n=== Processing {language}: {len(article_files)} articles ===")

        for idx, article_file in enumerate(article_files, 1):
            doc_id = article_file.stem  # filename without extension

            print(f"\n[{idx}/{len(article_files)}] Processing: {language}/{doc_id}")

            # Check if already processed
            output_file = output_lang_dir / f"{doc_id}.json"
            if output_file.exists():
                print(f"  Skipping - already exists")
                continue

            # Read article
            title, text = read_article(article_file)
            full_text = f"{title}\n\n{text}"

            # Check if article has enough content
            if len(full_text) < 500:
                print(f"  Skipping - insufficient content")
                continue

            print(f"  Article length: {len(full_text)} characters")

            # Generate questions using Claude
            result = generate_questions(full_text, title, doc_id, language)

            if result:
                # Save to file
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"  Saved {len(result['questions'])} questions to {output_file.name}")
                total_processed += 1
            else:
                print(f"  Failed to generate questions", result)
                time.sleep(60)

    print(f"\n Done! Processed {total_processed} articles total.")


if __name__ == "__main__":
    main()
