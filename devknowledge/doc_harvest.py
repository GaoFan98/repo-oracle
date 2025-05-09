import os
import ast
import re
from typing import Dict, List, Any, Optional, Tuple

import gitlab
import google.cloud.aiplatform as vertex_ai
from google import genai
from google.genai.types import HttpOptions
from google.genai.types import GenerationConfig
from google.genai.types import GenerateContentConfig

from .gitlab_util import (
    get_project,
    get_merge_request_changes,
    filter_python_files,
    create_branch,
    create_commit,
    create_merge_request
)

def init_vertex_ai():
    """Initialize Vertex AI client."""
    project = os.environ.get("GOOGLE_PROJECT")
    location = os.environ.get("GOOGLE_LOCATION", "us-central1")
    
    if not project:
        raise ValueError("GOOGLE_PROJECT environment variable not set")
    
    vertex_ai.init(project=project, location=location)

def extract_python_entities(file_content: str) -> List[Tuple[ast.AST, int, int]]:
    """Extract Python functions and classes that don't have docstrings."""
    try:
        tree = ast.parse(file_content)
        
        entities = []
        for node in ast.walk(tree):
            # Check for functions and classes
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                # Check if it has a docstring
                if not ast.get_docstring(node):
                    # Get the line range of the entity
                    start_line = node.lineno
                    end_line = node.body[-1].lineno if node.body else node.lineno
                    
                    entities.append((node, start_line, end_line))
        
        return entities
    except SyntaxError:
        # If the file has syntax errors, return an empty list
        return []

def get_entity_code(file_content: str, start_line: int, end_line: int) -> str:
    """Extract code for a specific entity."""
    lines = file_content.splitlines()
    return "\n".join(lines[start_line-1:end_line])

def generate_docstring(entity_code: str, entity_type: str, entity_name: str) -> str:
    """
    Generate a docstring for a Python entity using Vertex AI.
    Returns the generated docstring.
    """
    init_vertex_ai()
    
    # Configure the model parameters
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40
    }
    
    prompt = f"""
Add a concise Google-style docstring for the following Python {entity_type}:

```python
{entity_code}
```

The docstring should:
1. Summarize what the {entity_type} does
2. Document parameters (with types)
3. Document return values (with types)
4. Document exceptions that might be raised
5. Include examples if appropriate

Only return the docstring text without triple quotes.
"""
    
    # Call Vertex AI Codey to generate the docstring
    try:
        # Get project and location from environment variables
        project_id = os.environ.get("GOOGLE_PROJECT")
        location = os.environ.get("GOOGLE_LOCATION", "us-central1")
        
        if not project_id:
            print("Error: GOOGLE_PROJECT environment variable not set")
            return f"TODO: Add docstring for {entity_name}"
        
        # Set environment variables for Vertex AI integration
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        os.environ["GOOGLE_CLOUD_LOCATION"] = location
        
        # Create the Gen AI client with proper configuration
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        
        # Generate content using the Gen AI SDK
        response = client.models.generate_content(
            model="gemini-2.0-pro-001",
            contents=prompt,
            config=GenerateContentConfig(
                temperature=parameters["temperature"],
                max_output_tokens=parameters["max_output_tokens"],
                top_p=parameters["top_p"],
                top_k=parameters["top_k"]
            )
        )
        
        return response.text.strip()
    except Exception as e:
        # Fallback if the generation fails
        print(f"Error generating docstring: {str(e)}")
        return f"TODO: Add docstring for {entity_name}"

def insert_docstring(
    file_content: str, 
    entity: ast.AST, 
    docstring: str, 
    start_line: int
) -> str:
    """Insert a docstring into the file content at the appropriate position."""
    lines = file_content.splitlines()
    
    # Find the indentation level
    indent_match = re.match(r'(\s*)', lines[start_line-1])
    indent = indent_match.group(1) if indent_match else ""
    
    # Format the docstring
    docstring_lines = docstring.splitlines()
    formatted_docstring = f'{indent}    """\n'
    
    for line in docstring_lines:
        formatted_docstring += f'{indent}    {line}\n'
    
    formatted_docstring += f'{indent}    """\n'
    
    # Insert the docstring after the entity declaration
    header_line = lines[start_line-1]
    
    # Check if the entity's body starts on the same line as the header
    if ":" not in header_line:
        # This is a complex case, we'll need to find where the colon is
        i = start_line
        while i < len(lines) and ":" not in lines[i-1]:
            i += 1
        
        if i < len(lines):
            # Found the colon, insert after this line
            lines.insert(i, formatted_docstring)
        else:
            # Couldn't find the colon, insert after the header as a fallback
            lines.insert(start_line, formatted_docstring)
    else:
        # Simple case, insert after the header
        lines.insert(start_line, formatted_docstring)
    
    return "\n".join(lines)

def process_python_file(file_content: str) -> Tuple[str, bool]:
    """
    Process a Python file to add missing docstrings.
    Returns the updated file content and a boolean indicating whether any changes were made.
    """
    entities = extract_python_entities(file_content)
    
    if not entities:
        return file_content, False
    
    # Process entities in reverse order so that line numbers remain valid
    entities.sort(key=lambda x: x[1], reverse=True)
    
    updated_content = file_content
    made_changes = False
    
    for entity, start_line, end_line in entities:
        entity_type = "class" if isinstance(entity, ast.ClassDef) else "function"
        entity_name = entity.name
        
        entity_code = get_entity_code(file_content, start_line, end_line)
        docstring = generate_docstring(entity_code, entity_type, entity_name)
        
        # Insert the docstring
        updated_content = insert_docstring(updated_content, entity, docstring, start_line)
        made_changes = True
    
    return updated_content, made_changes

def process_merge_request(project_id: int, mr_iid: int):
    """
    Process a merge request to add docstrings to Python files.
    
    Args:
        project_id: The ID of the GitLab project
        mr_iid: The IID of the merge request
    """
    # Get the changes from the merge request
    changes = get_merge_request_changes(project_id, mr_iid)
    python_changes = filter_python_files(changes)
    
    if not python_changes:
        return
    
    # Create a branch for the AI docs
    branch_name = f"ai-docs/{mr_iid}"
    
    try:
        create_branch(project_id, branch_name)
    except gitlab.exceptions.GitlabCreateError:
        # Branch might already exist, continue anyway
        pass
    
    # Process each Python file
    commit_actions = []
    
    for change in python_changes:
        new_path = change["new_path"]
        new_content = change.get("diff", "")
        
        # Get the full file content using the GitLab API
        project = get_project(project_id)
        file_info = project.files.get(new_path, ref="main")
        file_content = file_info.decode().decode("utf-8")
        
        # Process the file
        updated_content, made_changes = process_python_file(file_content)
        
        if made_changes:
            commit_actions.append({
                "action": "update",
                "file_path": new_path,
                "content": updated_content
            })
    
    if commit_actions:
        # Create a commit with the changes
        create_commit(
            project_id,
            branch_name,
            f"Add AI-generated docstrings for MR !{mr_iid}",
            commit_actions
        )
        
        # Create a merge request
        create_merge_request(
            project_id,
            branch_name,
            "main",
            f"AI Docs for MR !{mr_iid}",
            "This MR adds AI-generated docstrings to Python files that were modified in the original MR.",
            ["ai-docs"]
        ) 