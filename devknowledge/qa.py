import os
from typing import Dict, List, Any
import google.cloud.aiplatform as vertex_ai

from .index import search_index, init_vertex_ai

def generate_answer(question: str) -> Dict[str, Any]:
    """
    Generate an answer to a question using Vertex AI.
    
    Args:
        question: The question to answer
    
    Returns:
        Dict with answer and citations
    """
    # Search for relevant snippets
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
    
    # Call Vertex AI to generate an answer
    init_vertex_ai()
    
    try:
        model_name = "text-bison"
        
        # Get the model
        model = vertex_ai.models.TextGenerationModel.from_pretrained(model_name)
        
        # Generate the answer
        response = model.predict(
            prompt=prompt,
            **parameters
        )
        
        answer_text = response.text.strip()
        
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