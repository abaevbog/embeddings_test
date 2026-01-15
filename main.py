import shutil
import torch
import json
import nltk
import gc
import time
from pathlib import Path
from eval_lib.model_eval import ModelEval
from config import DATASET, DOCUMENT_COUNT_TO_PROCESS, K_VALUES


# Download punkt tokenizer if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

MODELS_TO_TEST = [
    ModelEval(
        model_name="sentence-transformers/all-MiniLM-L6-v2", # Tiny model. Baseline. Ok on small corpus, only English
        reranker_model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    ),
    # Only model_name is required. This is just a full config example.
    ModelEval(
        model_name="BAAI/bge-large-en-v1.5", # Larger model, only English
        query_prefix="Represent this sentence for searching relevant passages: ", # Query prefix per for embeddings per model docs
        reranker_model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", # Optional reranker to apply after fetching embeddings
        #reranker_retrieval_k=30, # Optional number of top passages to rerank
        #chunk_size=512, # Size of chunks documents are split into for embedding (default: 512)
        #chunk_overlap=50, # By how many tokens chunks overlap (default: 50)
    ),
    ModelEval(
        model_name="nomic-ai/nomic-embed-text-v1", # Same as used by ZotSeek
        doc_prefix="search_document: ",
        query_prefix="search_query: ",
        reranker_model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    ),
    ModelEval(
        model_name="intfloat/multilingual-e5-large-instruct", # Good but slow
        query_prefix="Instruct: Given a query, retrieve relevant passages from the document.\nQuery: ",
        reranker_model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    ),
    ModelEval(
        model_name="Snowflake/snowflake-arctic-embed-l-v2.0", # Good but slow
        reranker_model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    ),
    ModelEval(
        model_name="BAAI/bge-m3", # Good but slow
        reranker_model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    ),
    # ModelEval(
    #     model_name="Qwen/Qwen3-Embedding-0.6B", # Supposed to be good - could not run locally
    # )
]

def main():
    # Clear output directory
    output_dir = Path(__file__).parent / "results"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
  
    # Load dataset
    dataset_path = Path(__file__).parent / "datasets" / DATASET / "data.json"
    data = open(dataset_path, 'r', encoding='utf-8')
    dataset = json.load(data)
    
    # Test on subset for faster iteration
    if DOCUMENT_COUNT_TO_PROCESS is not None:
        dataset = dataset[:DOCUMENT_COUNT_TO_PROCESS]
    
    
    # Test each model
    all_results = []
    for idx, model in enumerate(MODELS_TO_TEST):
        model.init(dataset)
        
        metrics = model.run()
        
        # Save results to file
        output_file = Path(__file__).parent / "results" / f"{idx}_{model.model_name.split('/')[-1]}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        all_results.append({ 'model_name': model.model_name, 'metrics': metrics })
        
        # Try to free up some memory
        model.uninit()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        time.sleep(5)  # Sleep a bit to hopefully free up the memory
    
    # Print summary showing metrics at each k value
    print(f"\n{'='*100}")
    print("SUMMARY: Evidence Coverage at Different K Values")
    print(f"{'='*100}")
    
    for k in K_VALUES:
        print(f"\n--- Results @{k} ---")
        print(f"{'Model':<40} {'Evid Cov':>10} {'Evid Rec':>10} {'Bin Recall':>11} {'Prec':>8} {'MRR':>8} {'NDCG':>8}")
        print(f"{'-'*97}")
        for result in all_results:
            model_name = result['model_name'].split('/')[-1][:35]
            metrics = result['metrics']
            evidence_coverage = metrics[f'evidence_coverage@{k}']
            evidence_recall = metrics[f'evidence_recall@{k}']
            chunk_recall = metrics[f'chunk_recall@{k}']
            precision = metrics[f'precision@{k}']
            mrr = metrics[f'mrr@{k}']
            ndcg = metrics[f'ndcg@{k}']
            print(f"{model_name:<40} {evidence_coverage*100:>9.1f}% {evidence_recall*100:>9.1f}% {chunk_recall*100:>10.1f}% {precision*100:>7.1f}% {mrr:>8.3f} {ndcg:>8.3f}")


if __name__ == "__main__":
    main()
