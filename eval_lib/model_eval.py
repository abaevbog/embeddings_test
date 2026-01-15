from typing import List, Dict, Any, Optional
import math
import gc
from eval_lib.rerank import CrossEncoderReranker
from .chunking import FixedTokensChunker, Chunker, is_relevant_chunk
from .embed import SentenceTransformerEmbedder, Embedder
from config import K_VALUES, LOG_DETAILED_RESULTS

class ModelEval:
    """
    Represents a model configuration for evaluation, including chunking and embedding settings.
    """
    def __init__(
        self,
        model_name: str,
        reranker_model_name: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        query_prefix: str = "",
        doc_prefix: str = "",
        reranker_retrieval_k: int = 30
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.reranker_model_name = reranker_model_name
        self.reranker_retrieval_k = reranker_retrieval_k or 30
        
        # Will be initialized when init() is called
        self.chunker: Optional[Chunker] = None
        self.embedder: Optional[Embedder] = None
        self.dataset: Optional[List[Dict[str, Any]]] = None
        self.reranker: Optional[CrossEncoderReranker] = None
        self.search_results: Optional[List[Dict[str, Any]]] = None
    
    def init(self, dataset: List[Dict[str, Any]]):
        """
        Initialize the chunker and embedder for this model configuration.
        
        Args:
            dataset: The dataset to prepare documents for chunking
        """
        print(f"Initializing {self.model_name}")
        print(f"  Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        
        # Store dataset first
        self.dataset = dataset
        
        # Create and prepare chunker
        print("  Preparing documents (chunking)...")
        self.chunker = FixedTokensChunker(chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        self.chunker.prepare_documents(dataset)
        
        # Create embedder (loads the model)
        print("  Loading embedding model...")
        self.embedder = SentenceTransformerEmbedder(
            self.model_name,
            query_prefix=self.query_prefix,
            doc_prefix=self.doc_prefix
        )
        print("  Model loaded and ready!")
        if self.reranker_model_name:
            print("  Loading reranker model...")
            self.reranker = CrossEncoderReranker(self.reranker_model_name)
            print("  Reranker model loaded and ready!")
    
    def uninit(self):
        del self.embedder
        self.embedder = None
        if self.reranker:
            del self.reranker
            self.reranker = None
        # Force garbage collection to free memory
        gc.collect()
    

    def run(self) -> Dict[str, Any]:
        print("Phase 1: Generating embeddings and building index...")
        self.embedder.embed_corpus(self.chunker.docs)
        
        print("Phase 2: Running search for all queries...")
        self.perform_search()
        
        print("Phase 3: Calculating metrics...")
        metrics = self.calculate_metrics()
        
        print("Phase 4: Formatting results...")
        if LOG_DETAILED_RESULTS:
            metrics['results'] = self.format_results()
        
        return metrics


    def perform_search(self):
        search_results = []
        
        # Count total queries for progress tracking
        total_queries = sum(len(doc['queries']) for doc in self.dataset)
        query_counter = 0
        
        for doc in self.dataset:
            doc_id = doc['doc_id']
            chunks = self.chunker.docs[doc_id]['chunks']
            
            for query_data in doc['queries']:
                query_id = query_data['query_id']
                query_text = query_data['query_text']
                relevant_matches = query_data['matches']
                
                # Search using embedder
                if self.reranker:
                    # Retrieve more candidates for reranking
                    k_to_retrieve = self.reranker_retrieval_k
                else:
                    # Retrieve enough to evaluate at all k values
                    max_k = max(K_VALUES)
                    k_to_retrieve = max(max_k, len(relevant_matches))
                
                search_result = self.embedder.search(
                    query_text, 
                    target_doc_id=doc_id, 
                    k=k_to_retrieve,
                )
                
                # Apply reranking if enabled
                if self.reranker and search_result.chunks_from_target_doc:
                    query_counter += 1
                    # Show progress every 10 queries or for first/last query
                    if query_counter == 1 or query_counter % 10 == 0 or query_counter == total_queries:
                        print(f" --- Reranking query {query_counter}/{total_queries}")
                    
                    # Extract passages from target doc for reranking
                    passages_with_indices = [
                        (chunk_idx, chunks[chunk_idx]) 
                        for chunk_idx, _ in search_result.chunks_from_target_doc
                    ]
                    
                    # Rerank and keep top results for evaluation
                    max_k = max(K_VALUES)
                    reranked = self.reranker.rerank_with_indices(
                        query_text, 
                        passages_with_indices, 
                        top_k=max(max_k, len(relevant_matches))
                    )
                    
                    # Update search_result with reranked chunks
                    search_result.chunks_from_target_doc = reranked
                
                search_results.append({
                    'doc_id': doc_id,
                    'query_id': query_id,
                    'query_text': query_text,
                    'relevant_matches': relevant_matches,
                    'search_result': search_result,
                    'chunks': chunks
                })
        
        self.search_results = search_results


    def calculate_metrics(self) -> Dict[str, Any]:
        total_queries = 0
        total_queries_with_matches = 0
        
        # Track metrics for each k value
        metrics_at_k = {}
        for k in K_VALUES:
            metrics_at_k[k] = {
                'correct_chunk_retrievals': 0,  # Binary: found ≥1 evidence
                'correct_doc_retrievals': 0,
                'total_precision': 0.0,
                'total_evidence_coverage': 0.0,
                'total_evidence_found': 0,
                'total_evidence_spans': 0,
                'total_reciprocal_rank': 0.0,  # MRR tracking
                'total_ndcg': 0.0  # NDCG tracking
            }
        
        for result in self.search_results:
            doc_id = result['doc_id']
            relevant_matches = result['relevant_matches']
            search_result = result['search_result']
            chunks = result['chunks']
            
            # Check chunk-level accuracy (only for queries with specific matches)
            if len(relevant_matches) > 0:
                total_queries_with_matches += 1
                num_evidence_spans = len(relevant_matches)
                
                # Evaluate at each k value
                for k in K_VALUES:
                    # Track which evidence spans were found at this k
                    evidence_found = set()
                    first_relevant_position = None  # Track position of first relevant chunk for MRR
                    relevance_at_position = []  # Track relevance (0 or 1) at each position for NDCG
                    
                    # Only evaluate the first k chunks
                    chunks_to_evaluate = search_result.chunks_from_target_doc[:k]
                    
                    for position, (chunk_idx, _) in enumerate(chunks_to_evaluate, start=1):
                        chunk_text = chunks[chunk_idx]
                        is_relevant = False
                        # Check if this chunk matches any of the expected matches
                        for evidence_idx, match in enumerate(relevant_matches):
                            if is_relevant_chunk(match['text'], chunk_text):
                                evidence_found.add(evidence_idx)
                                if first_relevant_position is None:
                                    first_relevant_position = position
                                is_relevant = True
                                break  # Count this chunk only once
                        
                        relevance_at_position.append(1 if is_relevant else 0)
                    
                    num_evidence_found = len(evidence_found)
                    
                    # Evidence coverage: what percentage of all evidence spans did we find?
                    evidence_coverage = num_evidence_found / num_evidence_spans if num_evidence_spans > 0 else 0
                    # Binary recall: did we find at least one evidence span?
                    if num_evidence_found > 0:
                        metrics_at_k[k]['correct_chunk_retrievals'] += 1
                    
                    metrics_at_k[k]['total_evidence_coverage'] += evidence_coverage
                    metrics_at_k[k]['total_evidence_spans'] += num_evidence_spans
                    metrics_at_k[k]['total_evidence_found'] += num_evidence_found
                    
                    # Calculate precision@k
                    if len(chunks_to_evaluate) > 0:
                        precision = num_evidence_found / len(chunks_to_evaluate)
                        metrics_at_k[k]['total_precision'] += precision
                    
                    # Calculate MRR (Mean Reciprocal Rank)
                    if first_relevant_position is not None:
                        reciprocal_rank = 1.0 / first_relevant_position
                        metrics_at_k[k]['total_reciprocal_rank'] += reciprocal_rank
                    
                    # Calculate NDCG (Normalized Discounted Cumulative Gain)
                    if len(relevance_at_position) > 0:
                        # Calculate DCG (Discounted Cumulative Gain)
                        dcg = sum(rel / math.log2(pos + 1) for pos, rel in enumerate(relevance_at_position, start=1) if rel > 0)
                        
                        # Calculate IDCG (Ideal DCG) - based on ideal scenario where all relevant chunks are in top positions
                        # The ideal case is having min(k, num_evidence_spans) relevant chunks in the top-k positions
                        num_relevant_at_k = min(k, num_evidence_spans)
                        ideal_relevance = [1] * num_relevant_at_k
                        idcg = sum(rel / math.log2(pos + 1) for pos, rel in enumerate(ideal_relevance, start=1))
                        
                        # Normalize
                        ndcg = dcg / idcg if idcg > 0 else 0.0
                        metrics_at_k[k]['total_ndcg'] += ndcg
                    
                    # Doc-level accuracy
                    doc_hit = doc_id in search_result.retrieved_doc_ids and len(chunks_to_evaluate) > 0
                    if doc_hit:
                        metrics_at_k[k]['correct_doc_retrievals'] += 1
            
            total_queries += 1
        
        # Calculate aggregate metrics for each k
        results = {
            'total_queries': total_queries,
            'total_queries_with_matches': total_queries_with_matches
        }
        
        for k in K_VALUES:
            m = metrics_at_k[k]
            chunk_recall = m['correct_chunk_retrievals'] / total_queries_with_matches if total_queries_with_matches > 0 else 0
            doc_recall = m['correct_doc_retrievals'] / total_queries_with_matches if total_queries_with_matches > 0 else 0
            avg_precision = m['total_precision'] / total_queries_with_matches if total_queries_with_matches > 0 else 0
            avg_evidence_coverage = m['total_evidence_coverage'] / total_queries_with_matches if total_queries_with_matches > 0 else 0
            overall_evidence_recall = m['total_evidence_found'] / m['total_evidence_spans'] if m['total_evidence_spans'] > 0 else 0
            mrr = m['total_reciprocal_rank'] / total_queries_with_matches if total_queries_with_matches > 0 else 0
            ndcg = m['total_ndcg'] / total_queries_with_matches if total_queries_with_matches > 0 else 0
            
            results[f'chunk_recall@{k}'] = chunk_recall
            results[f'evidence_coverage@{k}'] = avg_evidence_coverage
            results[f'evidence_recall@{k}'] = overall_evidence_recall
            results[f'precision@{k}'] = avg_precision
            results[f'doc_recall@{k}'] = doc_recall
            results[f'mrr@{k}'] = mrr
            results[f'ndcg@{k}'] = ndcg
        
        return results


    def format_results(self) -> List[Dict[str, Any]]:
        """Phase 4: Format search results into detailed output."""
        formatted_results = []
        
        for result in self.search_results:
            doc_id = result['doc_id']
            query_id = result['query_id']
            query_text = result['query_text']
            relevant_matches = result['relevant_matches']
            search_result = result['search_result']
            
            # Compute doc hit for this result
            doc_hit = doc_id in search_result.retrieved_doc_ids
            
            # Format retrieved chunks with visual indicators and similarity scores
            retrieved_chunks = []
            chunks_in_doc = self.chunker.docs[doc_id]['chunks']
            
            for global_idx in search_result.global_indices:
                retrieved_doc_id, chunk_idx_in_doc = self.embedder.chunk_to_doc_map[global_idx]
                similarity = search_result.similarities[global_idx]
                
                if retrieved_doc_id == doc_id:
                    if len(relevant_matches) > 0:
                        # Check if this chunk is relevant to any expected match
                        chunk_text = chunks_in_doc[chunk_idx_in_doc]
                        is_relevant = any(is_relevant_chunk(match['text'], chunk_text) for match in relevant_matches)
                        indicator = "✓ " if is_relevant else "❌ "
                    else:
                        indicator = "✓ "  # Correct document for doc-level query
                    retrieved_chunks.append(f"{indicator}INDEX:{chunk_idx_in_doc} ({similarity:.2f}) -- {chunks_in_doc[chunk_idx_in_doc]}")
                else:
                    # Chunk from wrong document
                    wrong_doc_chunk = self.chunker.docs[retrieved_doc_id]['chunks'][chunk_idx_in_doc]
                    retrieved_chunks.append(f"❌ DOC:{retrieved_doc_id} INDEX:{chunk_idx_in_doc} ({similarity:.2f}) -- {wrong_doc_chunk}")
            
            # Extract expected match texts
            expected_matches = [match['text'] for match in relevant_matches]
            
            formatted_results.append({
                'doc_id': doc_id,
                'query_id': query_id,
                'query_text': query_text,
                'expected_matches': expected_matches,
                'retrieved_chunks': retrieved_chunks,
                'doc_hit': doc_hit
            })
        
        return formatted_results