import os
import json
import glob
import faiss
import numpy as np
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import google.cloud.aiplatform as vertex_ai
from google import genai
from google.genai.types import HttpOptions
import subprocess

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
    # Skip .git directory files
    if "/.git/" in file_path:
        return []
    
    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    # Skip certain binary or non-text file types
    skip_extensions = ['.pyc', '.jar', '.war', '.zip', '.tar', '.gz', '.class', '.exe', '.dll', '.so', '.bin']
    if ext in skip_extensions:
        print(f"Skipping binary file: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with a different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            print(f"Processed file with latin-1 encoding: {file_path}")
        except Exception as e:
            print(f"Skipping file due to encoding issue: {file_path}")
            return []
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return []
    
    # Skip empty files
    if not content.strip():
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
    Generate embeddings for text using Google's Gen AI API.
    
    Args:
        text: The text to embed
        
    Returns:
        numpy array of embeddings
    """
    try:
        # Get project and location from environment variables
        project_id = os.environ.get("GOOGLE_PROJECT")
        location = os.environ.get("GOOGLE_LOCATION", "us-central1")
        
        if not project_id:
            print("Error: GOOGLE_PROJECT environment variable not set")
            return generate_fallback_embedding(text)
        
        # Set environment variables for Vertex AI integration
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        os.environ["GOOGLE_CLOUD_LOCATION"] = location
        
        # Create the Gen AI client with proper configuration
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        
        # Use the current embed_content method
        model = "text-embedding-004"  # Use a supported embedding model
        response = client.models.embed_content(
            model=model,
            contents=[text]
        )
        
        # Extract embedding from response
        if response and hasattr(response, 'embeddings') and len(response.embeddings) > 0:
            return np.array(response.embeddings[0].values, dtype=np.float32)
            
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
    
    # If we get here, generate a fallback embedding
    return generate_fallback_embedding(text)

def generate_fallback_embedding(text: str) -> np.ndarray:
    """Generate a consistent random embedding as a fallback."""
    print("Using fallback random embeddings")
    # Generate a random but consistent embedding based on the hash of the text
    import hashlib
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % 10000
    np.random.seed(seed)
    return np.random.randn(768).astype(np.float32)

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
    2. Processes all relevant files (not just Python and Markdown)
    3. Generates embeddings for chunks
    4. Builds a FAISS index
    5. Saves the index and metadata
    """
    # Clone the repository
    repo_path = clone_repository()
    
    try:
        # Get all files from the repository
        print(f"Indexing all relevant files in {repo_path}")
        
        # Find all text-based files that are useful to index
        # Include source code, config files, docs, etc.
        all_files_result = subprocess.run(
            [
                "find", repo_path, "-type", "f", 
                "-not", "-path", "*/\\.git/*",
                "-and", "(",
                "-name", "*.py", "-o",
                "-name", "*.md", "-o",
                "-name", "*.java", "-o",
                "-name", "*.kt", "-o", 
                "-name", "*.js", "-o",
                "-name", "*.html", "-o",
                "-name", "*.css", "-o",
                "-name", "*.yml", "-o", 
                "-name", "*.yaml", "-o",
                "-name", "*.xml", "-o",
                "-name", "*.json", "-o",
                "-name", "*.txt", "-o",
                "-name", "*.properties", "-o",
                "-name", "Dockerfile", "-o",
                "-name", "docker-compose.yml", "-o",
                "-name", "*.gradle", "-o",
                "-name", "*.sh",
                ")"
            ],
            check=True, capture_output=True, text=True
        )
        
        # Process the results safely to avoid backslash issues in f-strings
        all_files_lines = all_files_result.stdout.strip().split('\n')
        all_files = [f for f in all_files_lines if f]  # Filter out any empty strings
        
        print(f"Found {len(all_files)} text-based files to process")
        
        # Process files
        all_chunks = []
        
        for file_path in all_files:
            try:
                chunks = chunk_file(file_path)
                
                # Make file paths relative to the repository
                for chunk in chunks:
                    chunk.filepath = os.path.relpath(chunk.filepath, repo_path)
                
                all_chunks.extend(chunks)
                
                # Print progress for every 10 files
                if len(all_chunks) % 10 == 0:
                    print(f"Processed {len(all_chunks)} chunks so far...")
                
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
        
        print(f"Created total of {len(all_chunks)} chunks from {len(all_files)} files")
        
        # Generate embeddings
        vectors = []
        metadata = []
        
        print("Generating embeddings for chunks...")
        for i, chunk in enumerate(all_chunks):
            vector = embed_text(chunk.text)
            vectors.append(vector)
            metadata.append(asdict(chunk))
            
            # Print progress for every 20 chunks
            if i > 0 and i % 20 == 0:
                print(f"Generated embeddings for {i}/{len(all_chunks)} chunks")
        
        if not vectors:
            print("No vectors to index, aborting")
            return
        
        print(f"Successfully generated {len(vectors)} vectors")
        
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
            
        print(f"Indexing complete! Index saved to {index_path}")
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
        print(f"No index found, attempting to build index for query: {query}")
        reindex_repository()
        index, metadata = load_index()
        
        if index is None:
            # Still no index, return empty results
            print("Failed to build index, returning empty results")
            return []
    
    print(f"Searching index for query: {query}")
    print(f"Index contains {index.ntotal} vectors and metadata has {len(metadata)} entries")
    
    # Embed the query
    query_vector = embed_text(query)
    
    # Convert to numpy array and normalize
    query_vector = np.array([query_vector], dtype=np.float32)
    faiss.normalize_L2(query_vector)
    
    # Search the index
    distances, indices = index.search(query_vector, k)
    
    # Get the metadata for the top k chunks
    results = []
    print(f"Search returned {len(indices[0])} results:")
    for i, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(metadata):
            result = metadata[idx].copy()
            result["score"] = float(distances[0][i])
            print(f"  Result {i+1}: {result['filepath']} (score: {result['score']:.4f}), lines {result['start_line']}-{result['end_line']}")
            results.append(result)
        else:
            print(f"  Result {i+1}: Invalid index {idx}")
    
    return results

if __name__ == "__main__":
    reindex_repository() 