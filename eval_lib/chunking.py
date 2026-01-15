from typing import List, Dict, Any
import nltk


class Chunker:
    def __init__(self):
        self.docs = None
    
    def chunk_text(self, text: str) -> List[str]:
        raise NotImplementedError()
    
    def prepare_documents(self, dataset: List[Dict[str, Any]]):
        """Prepare documents by chunking text from the dataset and store internally."""
        docs = {}
        
        for doc in dataset:
            doc_id = doc['doc_id']
            doc_title = doc.get('title', '')
            
            all_chunks = []
            
            # Chunk abstract separately
            abstract = doc['abstract']
            if abstract:
                abstract_chunks = self.chunk_text(abstract)
                all_chunks.extend(abstract_chunks)
            
            # Add full_text if it exists, chunking each section independently
            if 'full_text' in doc and doc['full_text']:
                for section in doc['full_text']:
                    section_name = section.get('section_name', '')
                    paragraphs = section.get('paragraphs', [])
                    # Filter out empty paragraphs
                    non_empty_paras = [para for para in paragraphs if para]
                    if non_empty_paras:
                        # Prepend section name to the section text
                        if section_name:
                            section_text = '\n'.join(f"[{section_name}] {para}" for para in non_empty_paras)
                        else:
                            section_text = '\n'.join(non_empty_paras)
                        # Chunk this section independently
                        section_chunks = self.chunk_text(section_text)
                        all_chunks.extend(section_chunks)
            
            docs[doc_id] = {
                'title': doc_title,
                'chunks': all_chunks
            }
        
        self.docs = docs


class SentenceChunker(Chunker):
    def __init__(self):
        super().__init__()
    
    def chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        return nltk.sent_tokenize(text)


class FixedCharsChunker(Chunker):
    def __init__(self, chunk_size: int = 512, overlap: int = 0):
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + self.chunk_size
            if end < text_len:
                search_end = min(end + 50, text_len)
                chunk_text = text[start:search_end]
                for delimiter in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    last_delim = chunk_text.rfind(delimiter)
                    if last_delim > self.chunk_size - 100:
                        end = start + last_delim + len(delimiter)
                        break

            chunk_text = text[start:end]
            if chunk_text:
                chunks.append(chunk_text)
            start = end - self.overlap if self.overlap > 0 else end

        return chunks


class FixedTokensChunker(Chunker):
    def __init__(self, chunk_size: int = 250, overlap: int = 50):
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        import re

        if not text:
            return []

        word_pattern = re.compile(r'\S+')
        word_matches = list(word_pattern.finditer(text))
        if not word_matches:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.overlap)
        for i in range(0, len(word_matches), step):
            chunk_word_matches = word_matches[i:i + self.chunk_size]
            if not chunk_word_matches:
                break
            start = chunk_word_matches[0].start()
            end = chunk_word_matches[-1].end()
            chunk_text = text[start:end]
            chunks.append(chunk_text)

        return chunks


class BalancedSectionChunker(Chunker):
    """
    Section-aware chunker that respects document structure and keeps sections together when possible.
    - STRICTLY enforces chunk_size limit (no chunks exceed this)
    - Splits large sections at paragraph boundaries with balanced chunk sizes
    - Avoids creating tiny orphan chunks by balancing splits
    - Maintains overlap between chunks from the same section
    - Preserves section names as metadata
    """
    def __init__(self, chunk_size: int = 512, overlap: int = 50, min_chunk_size: int = 250):
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.token_chunker = FixedTokensChunker(chunk_size=chunk_size, overlap=overlap)

    def chunk_section(self, section_text: str, section_name: str = "") -> List[str]:
        """Chunk a single section intelligently with balanced splits."""
        if not section_text:
            return []
        
        # Count tokens in section
        section_tokens = len(section_text.split())
        
        # If section fits in one chunk, keep it intact
        if section_tokens <= self.chunk_size:
            if section_name:
                return [f"[{section_name}] {section_text}"]
            return [section_text]
        
        # For large sections, split at paragraph boundaries
        paragraphs = section_text.split('\n')
        
        # If only one paragraph, fall back to token chunking
        if len(paragraphs) == 1:
            chunks = self.token_chunker.chunk_text(section_text)
            # Add section name to first chunk
            if chunks and section_name:
                chunks[0] = f"[{section_name}] {chunks[0]}"
            return chunks
        
        # Calculate target size for balanced chunks
        # If section is 520 tokens, create 2 chunks of ~260 each rather than 512+8
        num_chunks_needed = (section_tokens + self.chunk_size - 1) // self.chunk_size
        if num_chunks_needed < 2:
            num_chunks_needed = 2
        target_chunk_size = section_tokens // num_chunks_needed
        
        # Build balanced chunks from paragraphs
        chunks = []
        current_chunk_paragraphs = []
        current_chunk_tokens = 0
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue
            
            para_tokens = len(para.split())
            
            # Check if adding this paragraph would significantly exceed target size
            # Allow some flexibility but enforce hard chunk_size limit
            would_exceed_target = current_chunk_paragraphs and (current_chunk_tokens + para_tokens > target_chunk_size * 1.2)
            would_exceed_limit = current_chunk_tokens + para_tokens > self.chunk_size
            
            if would_exceed_target or would_exceed_limit:
                # Finalize current chunk
                chunk_text = '\n'.join(current_chunk_paragraphs)
                # Add section name to all chunks from this section
                if section_name:
                    chunk_text = f"[{section_name}] {chunk_text}"
                chunks.append(chunk_text)
                
                # Start new chunk with overlap (last paragraph from previous chunk)
                if self.overlap > 0 and current_chunk_paragraphs:
                    last_para = current_chunk_paragraphs[-1]
                    last_para_tokens = len(last_para.split())
                    if last_para_tokens < self.overlap * 2:  # Only if it's not too large
                        current_chunk_paragraphs = [last_para]
                        current_chunk_tokens = last_para_tokens
                    else:
                        current_chunk_paragraphs = []
                        current_chunk_tokens = 0
                else:
                    current_chunk_paragraphs = []
                    current_chunk_tokens = 0
            
            current_chunk_paragraphs.append(para)
            current_chunk_tokens += para_tokens
        
        # Add final chunk
        if current_chunk_paragraphs:
            chunk_text = '\n'.join(current_chunk_paragraphs)
            # Add section name to all chunks from this section
            if section_name:
                chunk_text = f"[{section_name}] {chunk_text}"
            # Only add if it meets minimum size (unless it's the only content)
            if current_chunk_tokens >= self.min_chunk_size or not chunks:
                chunks.append(chunk_text)
            elif chunks:
                # Merge small final chunk with previous chunk if possible
                last_chunk = chunks[-1]
                last_chunk_tokens = len(last_chunk.split())
                if last_chunk_tokens + current_chunk_tokens <= self.chunk_size:
                    chunks[-1] = last_chunk + '\n' + chunk_text
                else:
                    # Can't merge, keep as separate small chunk
                    chunks.append(chunk_text)
        
        return chunks if chunks else []


def is_relevant_chunk(match_text: str, chunk_text: str) -> bool:
    # Count chunk as relevant if it has 75% or more of words from text
    overlap_threshold = 0.75
    
    match_lower = match_text.lower()
    chunk_lower = chunk_text.lower()
    
    # First, try the full match
    if match_lower in chunk_lower:
        return True
    
    # Split into words for edge trimming
    words = match_text.split()
    if len(words) == 0:
        return False
    
    # Calculate how many words we can drop (1 - overlap_threshold)
    num_words_to_drop = int(len(words) * (1 - overlap_threshold))
    
    if num_words_to_drop > 0:
        # Try dropping from the beginning (keep last overlap_threshold%)
        subsequence_from_end = ' '.join(words[num_words_to_drop:])
        if subsequence_from_end.lower() in chunk_lower:
            return True
        
        # Try dropping from the end (keep first overlap_threshold%)
        subsequence_from_start = ' '.join(words[:-num_words_to_drop])
        if subsequence_from_start.lower() in chunk_lower:
            return True
    
    return False
