import os
import json
import glob
import faiss
import numpy as np
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import google.cloud.aiplatform as vertex_ai

from .gitlab_util import clone_repository

@dataclass
class ChunkMetadata:
    """Metadata for a code or documentation chunk."""
    filepath: str
    start_line: int
    end_line: int
    text: str

def init_vertex_ai():
    """Initialize Vertex AI client."""
    project = os.environ.get("GOOGLE_PROJECT")
    location = os.environ.get("GOOGLE_LOCATION", "us-central1")
    
    if not project:
        raise ValueError("GOOGLE_PROJECT environment variable not set")
    
    vertex_ai.init(project=project, location=location)

def chunk_file(file_path: str, max_tokens: int = 2048) -> List[ChunkMetadata]:
    """
    Split a file into chunks of approximately max_tokens tokens.
    
    Args:
        file_path: Path to the file
        max_tokens: Maximum number of tokens per chunk (approximate)
        
    Returns:
        List of ChunkMetadata objects
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Skip binary files
        return []
    
    lines = content.splitlines()
    
    # Simple approximation: 1 token ≈ 4 characters
    tokens_per_line = [len(line) // 4 + 1 for line in lines]
    
    chunks = []
    current_chunk_lines = []
    current_tokens = 0
    start_line = 0
    
    for i, (line, tokens) in enumerate(zip(lines, tokens_per_line)):
        # If adding this line would exceed max_tokens, save the current chunk
        if current_tokens + tokens > max_tokens and current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines)
            chunks.append(ChunkMetadata(
                filepath=file_path,
                start_line=start_line + 1,  # 1-indexed
                end_line=start_line + len(current_chunk_lines),
                text=chunk_text
            ))
            
            current_chunk_lines = []
            current_tokens = 0
            start_line = i
        
        current_chunk_lines.append(line)
        current_tokens += tokens
    
    # Add the final chunk if there's any content left
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines)
        chunks.append(ChunkMetadata(
            filepath=file_path,
            start_line=start_line + 1,  # 1-indexed
            end_line=start_line + len(current_chunk_lines),
            text=chunk_text
        ))
    
    return chunks

def embed_text(text: str) -> np.ndarray:
    """
    Generate embeddings for text using Vertex AI.
    
    Args:
        text: The text to embed
        
    Returns:
        numpy array of embeddings
    """
    init_vertex_ai()
    
    try:
        model_name = "textembedding-gecko"
        
        # Call the model
        model = vertex_ai.models.TextEmbeddingModel.from_pretrained(model_name)
        embeddings = model.get_embeddings([text])
        
        if embeddings and embeddings[0].values:
            # Convert to numpy array
            return np.array(embeddings[0].values, dtype=np.float32)
        else:
            # Fallback to zeros if embedding fails
            return np.zeros(768, dtype=np.float32)
    except Exception as e:
        # Fallback to zeros if the API call fails
        return np.zeros(768, dtype=np.float32)

def create_index(vectors: np.ndarray) -> faiss.Index:
    """
    Create a FAISS index for vectors.
    
    Args:
        vectors: Numpy array of vectors to index
        
    Returns:
        FAISS index
    """
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(vectors)
    
    # Add vectors to the index
    index.add(vectors)
    
    return index

def ensure_data_dir() -> str:
    """
    Ensure the data directory exists and return its path.
    
    Returns:
        Path to the data directory
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def reindex_repository():
    """
    Reindex the repository.
    
    This function:
    1. Clones the repository
    2. Processes Python files and Markdown docs
    3. Generates embeddings for chunks
    4. Builds a FAISS index
    5. Saves the index and metadata
    """
    # Clone the repository
    repo_path = clone_repository()
    
    try:
        # Find Python files and Markdown docs
        python_files = glob.glob(os.path.join(repo_path, "**", "*.py"), recursive=True)
        markdown_files = glob.glob(os.path.join(repo_path, "**", "*.md"), recursive=True)
        
        # Process files
        all_files = python_files + markdown_files
        all_chunks = []
        
        for file_path in all_files:
            chunks = chunk_file(file_path)
            
            # Make file paths relative to the repository
            for chunk in chunks:
                chunk.filepath = os.path.relpath(chunk.filepath, repo_path)
            
            all_chunks.extend(chunks)
        
        # Generate embeddings
        vectors = []
        metadata = []
        
        for chunk in all_chunks:
            vector = embed_text(chunk.text)
            vectors.append(vector)
            metadata.append(asdict(chunk))
        
        if not vectors:
            # No vectors to index
            return
        
        # Convert to numpy array
        vectors_array = np.array(vectors, dtype=np.float32)
        
        # Create FAISS index
        index = create_index(vectors_array)
        
        # Save index and metadata
        data_dir = ensure_data_dir()
        index_path = os.path.join(data_dir, "vector.index")
        metadata_path = os.path.join(data_dir, "meta.json")
        
        # Save FAISS index
        faiss.write_index(index, index_path)
        
        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    finally:
        # Clean up
        if os.path.exists(repo_path):
            import shutil
            shutil.rmtree(repo_path)

def load_index() -> Tuple[Optional[faiss.Index], List[Dict[str, Any]]]:
    """
    Load the FAISS index and metadata.
    
    Returns:
        Tuple of (FAISS index, metadata list)
    """
    data_dir = ensure_data_dir()
    index_path = os.path.join(data_dir, "vector.index")
    metadata_path = os.path.join(data_dir, "meta.json")
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return None, []
    
    # Load FAISS index
    index = faiss.read_index(index_path)
    
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    return index, metadata

def search_index(query: str, k: int = 6) -> List[Dict[str, Any]]:
    """
    Search the index for similar chunks.
    
    Args:
        query: The search query
        k: Number of results to return
        
    Returns:
        List of metadata for the top k chunks
    """
    index, metadata = load_index()
    
    if index is None:
        # No index yet, try to build it
        reindex_repository()
        index, metadata = load_index()
        
        if index is None:
            # Still no index, return empty results
            return []
    
    # Embed the query
    query_vector = embed_text(query)
    
    # Convert to numpy array and normalize
    query_vector = np.array([query_vector], dtype=np.float32)
    faiss.normalize_L2(query_vector)
    
    # Search the index
    distances, indices = index.search(query_vector, k)
    
    # Get the metadata for the top k chunks
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(metadata):
            result = metadata[idx].copy()
            result["score"] = float(distances[0][i])
            results.append(result)
    
    return results

if __name__ == "__main__":
    reindex_repository() 