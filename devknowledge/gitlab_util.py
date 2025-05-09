import os
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Tuple
import gitlab
from gitlab.v4.objects import Project, MergeRequest
import shutil

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

def download_repository_files(temp_dir: str) -> None:
    """
    Download all repository files from all branches directly using the GitLab API.
    This ensures we get all files regardless of branch structure.
    """
    try:
        print("Fetching all repository files from GitLab API...")
        project = get_project()
        
        # Get all branches
        branches = project.branches.list(all=True)
        print(f"Found {len(branches)} branches: {', '.join([b.name for b in branches])}")
        
        # Process main branch first, then others
        branch_names = ['main'] + [b.name for b in branches if b.name != 'main']
        
        # Process each branch
        for branch_name in branch_names:
            print(f"Processing branch: {branch_name}")
            branch_dir = os.path.join(temp_dir, f"branch_{branch_name}")
            os.makedirs(branch_dir, exist_ok=True)
            
            try:
                # Get the repository tree for this branch
                items = project.repository_tree(ref=branch_name, recursive=True, all=True)
                
                # Download each file
                for item in items:
                    if item['type'] == 'blob':  # only process files, not directories
                        file_path = item['path']
                        try:
                            # Get file content
                            file_content = project.files.get(file_path=file_path, ref=branch_name)
                            
                            # Create directory structure
                            file_dir = os.path.dirname(os.path.join(branch_dir, file_path))
                            os.makedirs(file_dir, exist_ok=True)
                            
                            # Write file content
                            with open(os.path.join(branch_dir, file_path), 'wb') as f:
                                f.write(file_content.decode())
                                
                            print(f"Downloaded: {branch_name}/{file_path}")
                        except Exception as e:
                            print(f"Error downloading {branch_name}/{file_path}: {str(e)}")
            except Exception as e:
                print(f"Error processing branch {branch_name}: {str(e)}")
        
        # Now create a merged version for indexing
        merged_dir = os.path.join(temp_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        
        # Copy files from feature/fastapi branch first (if it exists)
        fastapi_branch_dir = os.path.join(temp_dir, "branch_feature/fastapi")
        if os.path.exists(fastapi_branch_dir):
            print("Prioritizing files from feature/fastapi branch...")
            shutil.copytree(fastapi_branch_dir, merged_dir, dirs_exist_ok=True)
        
        # Then copy files from main branch if they don't exist in the feature branch
        main_branch_dir = os.path.join(temp_dir, "branch_main")
        if os.path.exists(main_branch_dir):
            print("Adding missing files from main branch...")
            # Use rsync-like copying that doesn't overwrite existing files
            for root, dirs, files in os.walk(main_branch_dir):
                for file in files:
                    src_path = os.path.join(root, file)
                    rel_path = os.path.relpath(src_path, main_branch_dir)
                    dst_path = os.path.join(merged_dir, rel_path)
                    
                    if not os.path.exists(dst_path):
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        shutil.copy2(src_path, dst_path)
        
        print(f"Repository files downloaded and merged to {merged_dir}")
        
        # List all files in the merged directory
        all_files = []
        for root, dirs, files in os.walk(merged_dir):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(os.path.relpath(file_path, merged_dir))
        
        print(f"Total files in merged repository: {len(all_files)}")
        for file in all_files[:10]:  # Show first 10 files
            print(f"  - {file}")
        
        # Specifically check for Dockerfile
        dockerfile_path = os.path.join(merged_dir, "Dockerfile")
        if os.path.exists(dockerfile_path):
            print(f"Dockerfile found: {dockerfile_path}")
            with open(dockerfile_path, 'r') as f:
                print(f"Dockerfile contents: {f.read()[:200]}...")  # Show first 200 chars
        else:
            print("Dockerfile NOT found in merged repository")
    
    except Exception as e:
        print(f"Error downloading repository: {str(e)}")
        raise

def clone_repository() -> str:
    """Clone the repository and return the local path."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    try:
        # Download all files from all branches using GitLab API
        download_repository_files(temp_dir)
        
        # Return the merged directory for indexing
        return os.path.join(temp_dir, "merged")
    except Exception as e:
        print(f"Error in clone_repository: {str(e)}")
        # Fall back to traditional git clone if API method fails
        return git_clone_repository(temp_dir)

def git_clone_repository(base_dir: str) -> str:
    """Traditional git clone as a fallback method."""
    repo_url = os.environ.get("REPO_CLONE_URL")
    if not repo_url:
        raise ValueError("REPO_CLONE_URL environment variable not set")
    
    token = os.environ.get("GITLAB_TOKEN")
    if not token:
        raise ValueError("GITLAB_TOKEN environment variable not set")
    
    # Replace token placeholder if needed
    if "{token}" in repo_url:
        repo_url = repo_url.replace("{token}", token)
    
    # Create directory for git clone
    clone_dir = os.path.join(base_dir, "git_clone")
    os.makedirs(clone_dir, exist_ok=True)
    
    # Print debugging info
    print(f"Falling back to git clone from {repo_url.replace(token, '***TOKEN***')} to {clone_dir}")
    
    # Clone the repository with full history and all branches
    clone_result = subprocess.run(
        ["git", "clone", "--no-single-branch", "--depth=1000000", repo_url, clone_dir],
        capture_output=True,
        text=True
    )
    if clone_result.returncode != 0:
        print(f"Error cloning repository: {clone_result.stderr}")
        raise Exception("Git clone failed")
    
    print("Repository cloned successfully")
    
    # Fetch all branches
    fetch_result = subprocess.run(
        ["git", "-C", clone_dir, "fetch", "--all"],
        capture_output=True,
        text=True
    )
    if fetch_result.returncode != 0:
        print(f"Error fetching branches: {fetch_result.stderr}")
    
    # Check out the feature/fastapi branch specifically
    print("Checking out feature/fastapi branch...")
    checkout_result = subprocess.run(
        ["git", "-C", clone_dir, "checkout", "feature/fastapi"],
        capture_output=True,
        text=True
    )
    if checkout_result.returncode != 0:
        print(f"Error checking out feature/fastapi branch: {checkout_result.stderr}")
        print("Falling back to default branch...")
    else:
        print("Successfully checked out feature/fastapi branch")
    
    # List ALL files in the repository
    all_files = []
    for root, dirs, files in os.walk(clone_dir):
        if '.git' not in root:  # Skip .git directory
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(os.path.relpath(file_path, clone_dir))
    
    print(f"Found total of {len(all_files)} files in the git repository")
    
    return clone_dir

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