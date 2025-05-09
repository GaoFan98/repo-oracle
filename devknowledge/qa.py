import os
from typing import Dict, List, Any
import google.cloud.aiplatform as vertex_ai
from google import genai
from google.genai.types import HttpOptions
from google.genai.types import GenerationConfig
from google.genai.types import GenerateContentConfig
import re

from .index import search_index, init_vertex_ai, load_index

def generate_answer(question: str) -> Dict[str, Any]:
    """
    Generate an answer to a question using Vertex AI.
    
    Args:
        question: The question to answer
    
    Returns:
        Dict with answer and citations
    """
    # Check if this is a listing files query
    if re.search(r'list.*files|files.*list|show.*files|what.*files', question.lower()):
        print("File listing query detected")
        snippets = get_file_listing()
    # Check if this is a Docker or FastAPI-related query
    elif any(keyword.lower() in question.lower() for keyword in 
             ['docker', 'dockerfile', 'container', 'image', 'fastapi', 'api', 'rest', 'endpoint', 'python']):
        # Create a combined keyword list for both Docker and FastAPI
        keywords = ['docker', 'dockerfile', 'container', 'image', 'fastapi', 'api', 'rest', 'endpoint', 'python', 'analysis']
        snippets = keyword_search(keywords)
        if snippets:
            print(f"Found {len(snippets)} related files through keyword search")
        else:
            print("No related files found through keyword search")
    else:
        # Regular semantic search for other queries
        snippets = search_index(question, k=6)
    
    if not snippets:
        return {
            "answer": "I don't have enough information to answer that question. Try reindexing the repository first.",
            "citations": []
        }
    
    # Format the snippets for the prompt
    formatted_snippets = ""
    for i, snippet in enumerate(snippets):
        filepath = snippet["filepath"]
        start_line = snippet["start_line"]
        end_line = snippet["end_line"]
        text = snippet["text"]
        
        formatted_snippets += f"[Snippet {i+1}] {filepath} (lines {start_line}-{end_line}):\n{text}\n\n"
    
    # Prepare the prompt
    prompt = f"""You are a helpful software expert.
Using ONLY the snippets below, answer the developer's question.
Cite files and line numbers.

Snippets:
{formatted_snippets}

Question: {question}

Answer:"""
    
    # Configure model parameters
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 1024,
        "top_p": 0.8,
        "top_k": 40
    }
    
    # Initialize Vertex AI
    init_vertex_ai()
    
    try:
        # Get project and location from environment variables
        project_id = os.environ.get("GOOGLE_PROJECT")
        location = os.environ.get("GOOGLE_LOCATION", "us-central1")
        
        if not project_id:
            return {
                "answer": "Error: GOOGLE_PROJECT environment variable not set",
                "citations": []
            }
        
        # Set environment variables for Vertex AI integration
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        os.environ["GOOGLE_CLOUD_LOCATION"] = location
        
        # Create the Gen AI client with proper configuration
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        
        # Generate content using the Gen AI SDK
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt,
            config=GenerateContentConfig(
                temperature=parameters["temperature"],
                max_output_tokens=parameters["max_output_tokens"],
                top_p=parameters["top_p"],
                top_k=parameters["top_k"]
            )
        )
        
        answer_text = response.text
        
        # Extract citations from snippets for the response
        citations = []
        for snippet in snippets:
            citations.append({
                "filepath": snippet["filepath"],
                "start_line": snippet["start_line"],
                "end_line": snippet["end_line"]
            })
        
        return {
            "answer": answer_text,
            "citations": citations
        }
    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}",
            "citations": []
        }

def keyword_search(keywords: List[str]) -> List[Dict[str, Any]]:
    """
    Perform a direct keyword search for files in the metadata.
    This is a fallback for when embeddings are not working properly.
    
    Args:
        keywords: List of keywords to search for
    
    Returns:
        List of file snippets that match any of the keywords
    """
    # Load the index and metadata
    _, metadata = load_index()
    
    if not metadata:
        return []
    
    # Find files that match the keywords
    matching_files = []
    
    for item in metadata:
        filepath = item.get("filepath", "").lower()
        text = item.get("text", "").lower()
        
        # Check if the file contains any of the keywords
        if any(keyword.lower() in filepath or keyword.lower() in text for keyword in keywords):
            matching_files.append(item)
    
    return matching_files

def get_file_listing() -> List[Dict[str, Any]]:
    """
    Get a listing of all files in the project.
    
    Returns:
        List containing a single metadata item with the file listing
    """
    # Load the index and metadata
    _, metadata = load_index()
    
    if not metadata:
        return []
    
    # Create a set of unique filepaths
    unique_filepaths = set()
    for item in metadata:
        filepath = item.get("filepath", "")
        unique_filepaths.add(filepath)
    
    # Create a simple text representation of the file list
    file_list_text = "List of files in the project:\n\n"
    for filepath in sorted(unique_filepaths):
        file_list_text += f"* {filepath}\n"
    
    # Create a single metadata item with the file list
    file_listing = [{
        "filepath": "file_listing.txt",
        "start_line": 1,
        "end_line": len(unique_filepaths) + 1,
        "text": file_list_text
    }]
    
    return file_listing 