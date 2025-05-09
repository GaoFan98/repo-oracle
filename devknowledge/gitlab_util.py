import os
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Tuple
import gitlab
from gitlab.v4.objects import Project, MergeRequest

def get_gitlab_client() -> gitlab.Gitlab:
    """Create and return authenticated GitLab client."""
    token = os.environ.get("GITLAB_TOKEN")
    if not token:
        raise ValueError("GITLAB_TOKEN environment variable not set")
    
    host = os.environ.get("GITLAB_HOST", "https://gitlab.com")
    return gitlab.Gitlab(host, private_token=token)

def get_project(project_id: Optional[int] = None) -> Project:
    """Get GitLab project by ID."""
    if project_id is None:
        project_id_str = os.environ.get("PROJECT_ID")
        if not project_id_str:
            raise ValueError("PROJECT_ID environment variable not set")
        project_id = int(project_id_str)
    
    gl = get_gitlab_client()
    return gl.projects.get(project_id)

def get_merge_request(project_id: int, mr_iid: int) -> MergeRequest:
    """Get a merge request by its IID."""
    project = get_project(project_id)
    return project.mergerequests.get(mr_iid)

def get_merge_request_changes(project_id: int, mr_iid: int) -> List[Dict[str, Any]]:
    """Get the changes from a merge request."""
    mr = get_merge_request(project_id, mr_iid)
    changes = mr.changes()
    return changes["changes"]

def filter_python_files(changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter changes to include only Python files."""
    return [
        change for change in changes 
        if change["new_path"].endswith(".py") and change["new_file"] is False
    ]

def clone_repository() -> str:
    """Clone the repository and return the local path."""
    repo_url = os.environ.get("REPO_CLONE_URL")
    if not repo_url:
        raise ValueError("REPO_CLONE_URL environment variable not set")
    
    token = os.environ.get("GITLAB_TOKEN")
    if not token:
        raise ValueError("GITLAB_TOKEN environment variable not set")
    
    # Replace token placeholder if needed
    if "{token}" in repo_url:
        repo_url = repo_url.replace("{token}", token)
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Clone the repository
    subprocess.run(
        ["git", "clone", repo_url, temp_dir],
        check=True,
        capture_output=True
    )
    
    return temp_dir

def create_branch(project_id: int, branch_name: str, ref: str = "main") -> Dict[str, Any]:
    """Create a new branch in the repository."""
    project = get_project(project_id)
    return project.branches.create({
        "branch": branch_name,
        "ref": ref
    })

def create_commit(
    project_id: int, 
    branch: str, 
    commit_message: str, 
    actions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Create a commit in the repository."""
    project = get_project(project_id)
    return project.commits.create({
        "branch": branch,
        "commit_message": commit_message,
        "actions": actions
    })

def create_merge_request(
    project_id: int,
    source_branch: str,
    target_branch: str,
    title: str,
    description: str = "",
    labels: List[str] = None
) -> MergeRequest:
    """Create a merge request in the repository."""
    project = get_project(project_id)
    mr_params = {
        "source_branch": source_branch,
        "target_branch": target_branch,
        "title": title,
        "description": description,
    }
    
    if labels:
        mr_params["labels"] = labels
    
    return project.mergerequests.create(mr_params) 