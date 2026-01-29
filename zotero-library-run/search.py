import os
import json
import pickle
import numpy as np
from pathlib import Path

import boto3
from anthropic import Anthropic

runtime = boto3.client('sagemaker-runtime')

# Initialize Anthropic client
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    raise ValueError("Please set ANTHROPIC_API_KEY environment variable")
anthropic_client = Anthropic(api_key=API_KEY)

QUESTION = "How do various methods for word sense disambiguation compare in terms of their need for external knowledge resources?"
EXPECTED_ANSWER = "Based on the evidence, WSD methods vary significantly in their knowledge resource requirements. Supervised methods require substantial external resources, including training instances and linguistic knowledge representations, making them unsuitable for under-resourced languages that lack these resources. In contrast, completely unsupervised and knowledge-free approaches have been developed specifically to avoid dependence on external resources, motivated by issues like Zipfian distribution of training data and quality limitations of linguistic representations. Additionally, hybrid systems exist that aim to bridge these approaches by remaining completely unsupervised and knowledge-free while still providing the interpretability typically associated with knowledge-based systems."

# Files
EMBEDDINGS_FILE = Path(__file__).parent / ".embeddings"
OUTPUT_FILE = Path(__file__).parent / ".last_retrieved_chunks.json"

# Search config
TOP_K = 5


def load_embeddings() -> dict:
    """Load embeddings data from file."""
    with open(EMBEDDINGS_FILE, 'rb') as f:
        return pickle.load(f)


def embed_query(query: str, endpoint_name: str) -> np.ndarray:
    """Embed a single query using SageMaker endpoint."""
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps({"inputs": [query]})
    )
    result = json.loads(response['Body'].read().decode())
    return np.array(result[0], dtype=np.float32)


def search(query_embedding: np.ndarray, embeddings: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    """
    Find top-k most similar chunks using dot product similarity.

    Returns list of (chunk_index, similarity_score) tuples.
    """
    # Compute similarities (dot product for normalized embeddings)
    similarities = np.dot(embeddings, query_embedding)

    # Get top-k indices
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    return [(int(idx), float(similarities[idx])) for idx in top_k_indices]


def generate_answer(question: str, evidence_chunks: list[str]) -> str:
    """
    Use Anthropic API to generate an answer based on retrieved evidence.

    Returns the generated answer or "Not enough information" if evidence is insufficient.
    """
    evidence_text = "\n\n---\n\n".join(evidence_chunks)

    prompt = f"""You are answering a question using ONLY the evidence provided below.

QUESTION: {question}

EVIDENCE:
{evidence_text}

INSTRUCTIONS:
1. Answer the question using ONLY information found in the evidence above
2. If the evidence does not contain enough information to answer the question, respond with exactly: "Not enough information"
3. Be concise and direct in your answer
4. Do not make up or infer information not present in the evidence

ANSWER:"""

    message = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text.strip()


def validate_answer(generated_answer: str, expected_answer: str) -> str:
    """
    Use Anthropic API to check if generated answer matches expected answer.

    Returns "Correct" or "Wrong".
    """
    prompt = f"""Compare the following two answers and determine if the GENERATED ANSWER correctly captures the same information as the EXPECTED ANSWER.

EXPECTED ANSWER:
{expected_answer}

GENERATED ANSWER:
{generated_answer}

The generated answer does not need to be word-for-word identical. It should be considered correct if it conveys the same key information and meaning.

Respond with "Correct" if the answers match or "Wrong" if they do not. If answers do not match, add a brief explanation."""

    message = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text.strip()


def run_search(
    question: str,
    embeddings_data: dict,
    top_k: int = 5,
    expected_answer: str = None
) -> dict:
    """
    Run a single search query: embed question, find matches, generate answer, validate.

    Args:
        question: The question to search for
        embeddings_data: Dict containing embeddings, chunks, chunk_to_doc_map, item_metadata, endpoint_name
        top_k: Number of top matches to retrieve
        expected_answer: Optional expected answer for validation

    Returns:
        Dict with question, generated_answer, validation_result, top_k, results_by_title
    """
    embeddings = embeddings_data['embeddings']
    chunk_to_doc_map = embeddings_data['chunk_to_doc_map']
    chunks = embeddings_data['chunks']
    item_metadata = embeddings_data['item_metadata']
    endpoint_name = embeddings_data['endpoint_name']

    # Embed the question
    query_embedding = embed_query(question, endpoint_name)

    # Search for top matches
    results = search(query_embedding, embeddings, k=top_k)

    # Group results by document title
    results_by_title = {}
    for chunk_idx, score in results:
        item_key, chunk_num = chunk_to_doc_map[chunk_idx]
        title = item_metadata[item_key].get('title', item_key)
        chunk_text = chunks[chunk_idx]

        if title not in results_by_title:
            results_by_title[title] = {
                'item_key': item_key,
                'chunks': []
            }

        results_by_title[title]['chunks'].append({
            'score': round(score, 4),
            'chunk_index': chunk_num,
            'text': f"[Score: {score:.4f}] {chunk_text}"
        })

    # Sort chunks within each title by score
    for title_data in results_by_title.values():
        title_data['chunks'].sort(key=lambda x: x['score'], reverse=True)

    # Collect raw chunk texts for answer generation (without score prefix)
    evidence_chunks = [chunks[chunk_idx] for chunk_idx, _ in results]

    # Generate answer using Anthropic
    generated_answer = generate_answer(question, evidence_chunks)

    # Validate answer if expected answer is provided
    validation_result = None
    if expected_answer:
        validation_result = validate_answer(generated_answer, expected_answer)

    return {
        'question': question,
        'expected_answer': expected_answer,
        'generated_answer': generated_answer,
        'validation_result': validation_result,
        'top_k': top_k,
        'results_by_title': results_by_title
    }


def main():
    print(f"Question: {QUESTION}")
    print(f"Loading embeddings from: {EMBEDDINGS_FILE}")

    if not EMBEDDINGS_FILE.exists():
        print("Error: Embeddings file not found. Run make_embeddings.py first.")
        return

    embeddings_data = load_embeddings()
    print(f"Loaded {len(embeddings_data['chunks'])} chunks from {len(embeddings_data['item_metadata'])} documents")
    print(f"Endpoint: {embeddings_data['endpoint_name']}")

    # Run search
    print("\nRunning search...")
    result = run_search(
        question=QUESTION,
        embeddings_data=embeddings_data,
        top_k=TOP_K,
        expected_answer=EXPECTED_ANSWER
    )

    # Save to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {OUTPUT_FILE}")

    # Print generated answer
    print("\n" + "="*60)
    print("GENERATED ANSWER")
    print("="*60)
    print(result['generated_answer'])

    # Print validation result
    if result['validation_result']:
        print("\n" + "="*60)
        print(f"VALIDATION: {result['validation_result']}")
        print("="*60)


if __name__ == "__main__":
    main()