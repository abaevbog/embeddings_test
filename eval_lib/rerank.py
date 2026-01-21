from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder
import numpy as np


class Reranker:
    """Base class for rerankers that score query-passage pairs."""
    
    def rerank(self, query: str, passages: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Rerank passages for a given query.
        
        Args:
            query: The search query
            passages: List of passage texts to rerank
            top_k: Number of top results to return
            
        Returns:
            List of (original_index, score) tuples, sorted by score descending
        """
        raise NotImplementedError()


class CrossEncoderReranker(Reranker):
    """Reranker using CrossEncoder models that score query-passage pairs."""
    
    def __init__(self, model_name: str):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name or path
        """
        self.model_name = model_name
        self.model = CrossEncoder(model_name, trust_remote_code=True)
        print(f"Loaded reranker: {model_name}")
    
    def rerank(self, query: str, passages: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Rerank passages using cross-encoder scoring.
        
        Args:
            query: The search query
            passages: List of passage texts to rerank
            top_k: Number of top results to return
            
        Returns:
            List of (original_index, score) tuples, sorted by score descending
        """
        if not passages:
            return []
        
        # Prepare query-passage pairs
        pairs = [[query, passage] for passage in passages]
        
        # Score all pairs
        scores = self.model.predict(pairs, convert_to_numpy=True)
        
        # Create (index, score) tuples
        indexed_scores = [(idx, float(score)) for idx, score in enumerate(scores)]
        
        # Sort by score descending
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return indexed_scores[:top_k]
    
    def rerank_with_indices(
        self, 
        query: str, 
        passages_with_indices: List[Tuple[int, str]], 
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Rerank passages that already have indices attached.
        
        Args:
            query: The search query
            passages_with_indices: List of (original_index, passage_text) tuples
            top_k: Number of top results to return
            
        Returns:
            List of (original_index, score) tuples, sorted by score descending
        """
        if not passages_with_indices:
            return []
        
        # Extract passages and indices
        indices = [idx for idx, _ in passages_with_indices]
        passages = [passage for _, passage in passages_with_indices]
        
        # Prepare query-passage pairs
        pairs = [[query, passage] for passage in passages]
        
        # Score all pairs
        scores = self.model.predict(pairs, convert_to_numpy=True)
        
        # Create (original_index, score) tuples
        indexed_scores = [(indices[i], float(scores[i])) for i in range(len(scores))]
        
        # Sort by score descending
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return indexed_scores[:top_k]

class SagemakerReranker:
    """Reranker that uses a SageMaker endpoint for reranking."""
    
    def __init__(self, endpoint_name: str):
        """
        Initialize the SageMaker reranker.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint hosting the reranker model
        """
        self.endpoint_name = endpoint_name
        self.runtime = boto3.client('sagemaker-runtime')
    
    def rerank_with_indices(
        self, 
        query: str, 
        passages_with_indices: List[Tuple[int, str]], 
        top_k: int
    ) -> List[Tuple[int, float]]:
        if not passages_with_indices:
            return []
        
        # Extract indices and texts
        indices = [idx for idx, _ in passages_with_indices]
        texts = [text for _, text in passages_with_indices]
        
        # Call SageMaker endpoint
        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps({
                "query": query,
                "texts": texts
            })
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        
        # Result is a list of dicts with 'index' (position in input) and 'score'
        # Already sorted by score descending
        indexed_scores = []
        for item in result:
            # Get the original position in the input array
            input_position = item.get('index')
            score = float(item.get('score', 0.0))
            
            # Map back to our chunk index using the input position
            if input_position is not None and input_position < len(indices):
                chunk_idx = indices[input_position]
                indexed_scores.append((chunk_idx, score))
        
        # Results are already sorted by the endpoint, just take top_k
        return indexed_scores[:top_k]


