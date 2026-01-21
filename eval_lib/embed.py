from typing import List, Dict, Tuple, Set
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import gc
import time
import psutil
import os
import pickle
import json
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import SEARCH_ALL_CHUNKS

runtime = boto3.client('sagemaker-runtime')


class SearchResult:
    """Result of a search query."""
    def __init__(self, global_indices: List[int], similarities: Dict[int, float], 
                 chunks_from_target_doc: List[Tuple[int, float]], retrieved_doc_ids: Set[str]):
        self.global_indices = global_indices  # All retrieved chunk global indices
        self.similarities = similarities  # Map: global_idx -> similarity score
        self.chunks_from_target_doc = chunks_from_target_doc  # List of (chunk_idx_in_doc, score) for target doc
        self.retrieved_doc_ids = retrieved_doc_ids  # Set of doc_ids that were retrieved


class Embedder:
    def __init__(self, model_name: str, query_prefix: str = "", doc_prefix: str = ""):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.embeddings = None
        self.chunk_to_doc_map = None
        
    def embed_corpus(self, docs: Dict[str, Dict[str, any]]):
        """Embed all chunks across documents and build internal index."""
        raise NotImplementedError()
    

    def embed_query(self, query_text: str) -> np.ndarray:
        """Embed a single query with the appropriate query prefix."""
        return self.embed_with_prefix([query_text], self.query_prefix)[0]

    def search(self, query: str, target_doc_id: str, k: int = 3) -> SearchResult:
        """Search for top-k most similar chunks using dot product similarity."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return SearchResult([], {}, [], set())
        
        # Embed the query
        query_embedding = self.embed_query(query)
        # Compute similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        if SEARCH_ALL_CHUNKS:
            # Search globally across all documents
            k_actual = min(k, len(similarities))
            top_k_indices = np.argsort(similarities)[-k_actual:][::-1]
            
            global_indices = list(top_k_indices)
            similarity_dict = {int(idx): float(similarities[idx]) for idx in top_k_indices}
            
            # Filter to target document
            chunks_from_target = []
            retrieved_doc_ids = set()
            for global_idx in top_k_indices:
                doc_id, chunk_idx_in_doc = self.chunk_to_doc_map[global_idx]
                retrieved_doc_ids.add(doc_id)
                if doc_id == target_doc_id:
                    chunks_from_target.append((chunk_idx_in_doc, float(similarities[global_idx])))
        else:
            # Search only within target document
            doc_indices = [i for i, (d_id, _) in enumerate(self.chunk_to_doc_map) if d_id == target_doc_id]
            if not doc_indices:
                return SearchResult([], {}, [], set())
            
            doc_similarities = similarities[doc_indices]
            k_actual = min(k, len(doc_indices))
            top_k_local_indices = np.argsort(doc_similarities)[-k_actual:][::-1]
            
            # Map back to global indices
            global_indices = [doc_indices[i] for i in top_k_local_indices]
            similarity_dict = {doc_indices[i]: float(doc_similarities[i]) for i in top_k_local_indices}
            
            # All results are from target document
            chunks_from_target = [(self.chunk_to_doc_map[global_idx][1], float(similarities[global_idx])) 
                                 for global_idx in global_indices]
            retrieved_doc_ids = {target_doc_id}
        
        return SearchResult(global_indices, similarity_dict, chunks_from_target, retrieved_doc_ids)
    
    def _save_embeddings_to_disk(self):
        """Save embeddings and metadata to disk in embeddings_cache folder."""
        # Create embeddings_cache folder if it doesn't exist
        cache_dir = "embeddings_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create a safe filename from model name
        safe_model_name = self.model_name.replace("/", "_").replace("\\", "_")
        cache_file = os.path.join(cache_dir, f"{safe_model_name}.pkl")
        
        # Save embeddings and metadata
        cache_data = {
            'embeddings': self.embeddings,
            'chunk_to_doc_map': self.chunk_to_doc_map,
            'model_name': self.model_name
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Saved embeddings to {cache_file}")


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, query_prefix: str = "", doc_prefix: str = ""):
        super().__init__(model_name, query_prefix, doc_prefix)
        
        device = "cpu"
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    
    def embed_with_prefix(self, texts: List[str], prefix: str) -> np.ndarray:
        """Embed texts with a specific prefix."""
        if not texts:
            return np.array([])
        prefixed = [f"{prefix}{t}" for t in texts]
        
        embeddings = self.model.encode(prefixed, normalize_embeddings=True)
        return embeddings
    
    def embed_corpus(self, docs: Dict[str, Dict[str, any]]):
        """Embed all chunks across documents and build internal index.

        - docs: mapping doc_id -> {'title': str, 'chunks': List[str]}
        """
        chunks_count = 0
        all_embeddings = []
        chunk_to_doc_map = []
        
        total_docs = len(docs)
        doc_counter = 0
        
        # Convert dict to list
        doc_items = list(docs.items())
        
        # Process documents sequentially for local model
        for doc_id, doc_data in doc_items:
            chunks = doc_data.get('chunks', [])
            if not chunks:
                continue
            
            # Embed this document's chunks (title already included during chunking)
            doc_embeddings = self.embed_with_prefix(chunks, self.doc_prefix)
            doc_counter += 1
            
            if doc_embeddings is not None:
                all_embeddings.append(doc_embeddings)
                
                # Track metadata
                for chunk_idx in range(len(chunks)):
                    chunks_count += 1
                    chunk_to_doc_map.append((doc_id, chunk_idx))
            
            print(f" --- {doc_counter}/{total_docs} docs indexed ({chunks_count} chunks)", end='\r')
        
        if all_embeddings:
            self.embeddings = np.vstack(all_embeddings)
        else:
            self.embeddings = np.zeros((0, 0))
        
        self.chunk_to_doc_map = chunk_to_doc_map

        print(f"\nTotal chunks across all documents: {chunks_count}")
        
        # Save embeddings to disk
        self._save_embeddings_to_disk()


class SagemakerEmbedder(Embedder):
    def __init__(self, model_name: str, query_prefix: str = "", doc_prefix: str = ""):
        super().__init__(model_name, query_prefix, doc_prefix)
    
    def embed_with_prefix(self, texts: List[str], prefix: str) -> np.ndarray:
        """Embed texts with a specific prefix."""
        if not texts:
            return np.array([])
        prefixed = [f"{prefix}{t}" for t in texts]
        
        # TEI has a batch size limit of 32, so batch requests
        MAX_BATCH_SIZE = 32
        all_embeddings = []
        
        for i in range(0, len(prefixed), MAX_BATCH_SIZE):
            batch = prefixed[i:i + MAX_BATCH_SIZE]
            response = runtime.invoke_endpoint(
                EndpointName=self.model_name.replace('/', '-').replace('.', '-') + '-endpoint',
                ContentType='application/json',
                Body=json.dumps({"inputs": batch})
            )
            result = json.loads(response['Body'].read().decode())
            batch_embeddings = np.array(result, dtype=np.float32)
            all_embeddings.append(batch_embeddings)
        
        # Combine all batches
        embeddings = np.vstack(all_embeddings)
        
        # Defensive check for None values
        if embeddings.size == 0 or np.any(embeddings == None):
            raise ValueError(f"Invalid embeddings received from endpoint. Shape: {embeddings.shape}, contains None: {np.any(embeddings == None)}")
        
        return embeddings
    
    def embed_query(self, query_text: str) -> np.ndarray:
        """Embed a single query with the appropriate query prefix."""
        return self.embed_with_prefix([query_text], self.query_prefix)[0]

    def embed_corpus(self, docs: Dict[str, Dict[str, any]]):
        """Embed all chunks across documents and build internal index.

        - docs: mapping doc_id -> {'title': str, 'chunks': List[str]}
        """
        chunks_count = 0
        all_embeddings = []
        chunk_to_doc_map = []
        
        total_docs = len(docs)
        doc_counter = 0
        
        # Convert dict to list
        doc_items = list(docs.items())
        
        # Process documents in parallel batches for SageMaker
        BATCH_SIZE = 10
        
        for batch_start in range(0, len(doc_items), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(doc_items))
            batch = doc_items[batch_start:batch_end]
            
            print(f"\n[BATCH] Processing documents {batch_start+1}-{batch_end} of {total_docs}")
            
            def process_document(doc_id, doc_data):
                """Process a single document's chunks."""
                chunks = doc_data.get('chunks', [])
                if not chunks:
                    return doc_id, None, []
                
                # Embed this document's chunks (title already included during chunking)
                doc_embeddings = self.embed_with_prefix(chunks, self.doc_prefix)
                return doc_id, doc_embeddings, chunks
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                futures = [executor.submit(process_document, doc_id, doc_data) for doc_id, doc_data in batch]
                
                # Collect results in order
                for future in futures:
                    doc_id, doc_embeddings, chunks = future.result()
                    doc_counter += 1
                    
                    if doc_embeddings is not None:
                        all_embeddings.append(doc_embeddings)
                        
                        # Track metadata
                        for chunk_idx in range(len(chunks)):
                            chunks_count += 1
                            chunk_to_doc_map.append((doc_id, chunk_idx))
                    
                    print(f" --- {doc_counter}/{total_docs} docs indexed ({chunks_count} chunks)", end='\r')
    
        if all_embeddings:
            self.embeddings = np.vstack(all_embeddings)
        else:
            self.embeddings = np.zeros((0, 0))
        
        self.chunk_to_doc_map = chunk_to_doc_map

        print(f"\nTotal chunks across all documents: {chunks_count}")
        
        # Save embeddings to disk
        self._save_embeddings_to_disk()
    
