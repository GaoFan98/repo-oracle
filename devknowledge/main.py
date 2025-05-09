import os
import json
import hmac
import hashlib
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Response, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates

from .doc_harvest import process_merge_request
from .index import reindex_repository
from .qa import generate_answer

app = FastAPI(title="Repo Oracle")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static files directory if it doesn't exist
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=static_dir)

class QuestionRequest(BaseModel):
    question: str

def verify_gitlab_signature(payload_body: bytes, request: Request) -> bool:
    """Verify GitLab webhook signature."""
    if "X-Gitlab-Token" not in request.headers:
        return False
    
    webhook_token = request.headers["X-Gitlab-Token"]
    expected_token = os.environ.get("WEBHOOK_SECRET")
    
    if not expected_token:
        raise HTTPException(status_code=500, detail="WEBHOOK_SECRET not configured")
    
    return hmac.compare_digest(webhook_token, expected_token)

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    """GitLab webhook endpoint for processing merge requests."""
    payload_body = await request.body()
    
    if not verify_gitlab_signature(payload_body, request):
        raise HTTPException(status_code=403, detail="Invalid webhook signature")
    
    payload = json.loads(payload_body)
    event_type = request.headers.get("X-Gitlab-Event")
    
    # Handle merge request events
    if event_type == "Merge Request Hook":
        mr_state = payload.get("object_attributes", {}).get("state")
        target_branch = payload.get("object_attributes", {}).get("target_branch")
        
        if mr_state == "merged" and target_branch == "main":
            mr_iid = payload.get("object_attributes", {}).get("iid")
            project_id = payload.get("project", {}).get("id") or os.environ.get("PROJECT_ID")
            
            # Process the merge request in the background
            background_tasks.add_task(
                process_merge_request, 
                project_id=project_id, 
                mr_iid=mr_iid
            )
            
            return {"status": "processing", "merge_request_iid": mr_iid}
    
    return {"status": "ignored"}

@app.post("/reindex")
async def reindex(background_tasks: BackgroundTasks):
    """
    Reindex the repository to update the knowledge base.
    This runs in the background to avoid blocking the API.
    """
    # Start reindexing in the background
    background_tasks.add_task(reindex_repository)
    
    return {"status": "success", "message": "Reindexing started in the background. This may take a few minutes."}

@app.post("/ask", response_model=Dict[str, Any])
async def ask(question_request: QuestionRequest):
    """Endpoint to ask questions about the codebase."""
    try:
        answer = generate_answer(question_request.question)
        return answer
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"error": f"Failed to generate answer: {str(e)}"}
        )

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the frontend UI."""
    return templates.TemplateResponse("index.html", {"request": request}) 