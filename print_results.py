import json
from pathlib import Path
from config import K_VALUES


def main():
    results_dir = Path(__file__).parent / "results"
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # Read all JSON files
    all_results = []
    for json_file in sorted(results_dir.glob("*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                
            # Extract model name from filename (format: {idx}_{model_name}.json)
            filename = json_file.stem  # Remove .json extension
            # Remove index prefix (e.g., "0_all-MiniLM-L6-v2" -> "all-MiniLM-L6-v2")
            parts = filename.split('_', 1)
            model_name = parts[1] if len(parts) > 1 else parts[0]
            
            all_results.append({
                'model_name': model_name,
                'metrics': metrics
            })
        except Exception as e:
            print(f"Warning: Could not read {json_file}: {e}")
    
    if not all_results:
        print(f"No results found in {results_dir}")
        return
    
    # Print summary table in Markdown format
    print(f"\n# Evaluation Results Summary\n")
    print(f"| K | Model | Evid Cov | Evid Rec | Bin Recall | Prec | MRR | NDCG |")
    print(f"|---|-------|----------|----------|------------|------|-----|------|")
    
    for k in K_VALUES:
        for result in all_results:
            model_name = result['model_name'][:35]
            metrics = result['metrics']
            evidence_coverage = metrics[f'evidence_coverage@{k}']
            evidence_recall = metrics[f'evidence_recall@{k}']
            chunk_recall = metrics[f'chunk_recall@{k}']
            precision = metrics[f'precision@{k}']
            mrr = metrics[f'mrr@{k}']
            ndcg = metrics[f'ndcg@{k}']
            print(f"| {k} | {model_name} | {evidence_coverage*100:.1f}% | {evidence_recall*100:.1f}% | {chunk_recall*100:.1f}% | {precision*100:.1f}% | {mrr:.3f} | {ndcg:.3f} |")


if __name__ == "__main__":
    main()
