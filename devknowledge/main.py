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

from .doc_harvest import process_merge_request
from .index import reindex_repository
from .qa import generate_answer

app = FastAPI(title="DevKnowledge Hub")

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
    """Endpoint to trigger a reindex of the repository."""
    background_tasks.add_task(reindex_repository)
    return {"status": "reindexing_started"}

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
async def root():
    """Serve the frontend UI."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DevKnowledge Hub</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }
            .container {
                margin-top: 20px;
            }
            .chat-container {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 20px;
                max-height: 500px;
                overflow-y: auto;
                background-color: #f9f9f9;
                margin-bottom: 20px;
            }
            .chat-box {
                min-height: 100px;
            }
            .input-container {
                display: flex;
                margin-top: 20px;
            }
            #question-input {
                flex: 1;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 16px;
            }
            button {
                padding: 10px 20px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                margin-left: 10px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #2980b9;
            }
            .question {
                background-color: #e1f0fa;
                padding: 10px 15px;
                border-radius: 5px;
                margin: 10px 0;
                align-self: flex-end;
            }
            .answer {
                background-color: #f5f5f5;
                padding: 10px 15px;
                border-radius: 5px;
                margin: 10px 0;
            }
            .citation {
                font-size: 12px;
                color: #666;
                margin-top: 5px;
            }
            pre {
                background-color: #f0f0f0;
                padding: 10px;
                border-radius: 4px;
                overflow-x: auto;
            }
            code {
                font-family: monospace;
            }
        </style>
    </head>
    <body>
        <h1>DevKnowledge Hub</h1>
        <div class="container">
            <h2>Ask about your codebase</h2>
            <p>Type a question about your codebase to get AI-powered answers with source code references.</p>
            
            <div class="chat-container">
                <div id="chat-box" class="chat-box"></div>
            </div>
            
            <div class="input-container">
                <input type="text" id="question-input" placeholder="Ask a question about your codebase...">
                <button id="ask-button">Ask</button>
            </div>
        </div>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const chatBox = document.getElementById('chat-box');
                const questionInput = document.getElementById('question-input');
                const askButton = document.getElementById('ask-button');
                
                askButton.addEventListener('click', askQuestion);
                questionInput.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        askQuestion();
                    }
                });
                
                function askQuestion() {
                    const question = questionInput.value.trim();
                    if (!question) return;
                    
                    // Add question to chat
                    const questionDiv = document.createElement('div');
                    questionDiv.className = 'question';
                    questionDiv.textContent = question;
                    chatBox.appendChild(questionDiv);
                    
                    // Clear input
                    questionInput.value = '';
                    
                    // Disable button while processing
                    askButton.disabled = true;
                    askButton.textContent = 'Thinking...';
                    
                    // Send request to backend
                    fetch('/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ question: question })
                    })
                    .then(response => response.json())
                    .then(data => {
                        // Add answer to chat
                        const answerDiv = document.createElement('div');
                        answerDiv.className = 'answer';
                        
                        // Format the answer with markdown-like syntax
                        let formattedAnswer = data.answer;
                        
                        // Replace code blocks
                        formattedAnswer = formattedAnswer.replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>');
                        
                        // Replace inline code
                        formattedAnswer = formattedAnswer.replace(/`([^`]+)`/g, '<code>$1</code>');
                        
                        answerDiv.innerHTML = formattedAnswer;
                        
                        // Add citations if available
                        if (data.citations && data.citations.length > 0) {
                            const citationsDiv = document.createElement('div');
                            citationsDiv.className = 'citation';
                            citationsDiv.innerHTML = '<strong>Sources:</strong><br>';
                            
                            data.citations.forEach(citation => {
                                citationsDiv.innerHTML += `${citation.filepath} (lines ${citation.start_line}-${citation.end_line})<br>`;
                            });
                            
                            answerDiv.appendChild(citationsDiv);
                        }
                        
                        chatBox.appendChild(answerDiv);
                        
                        // Scroll to bottom
                        chatBox.scrollTop = chatBox.scrollHeight;
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        
                        // Add error message to chat
                        const errorDiv = document.createElement('div');
                        errorDiv.className = 'answer';
                        errorDiv.textContent = 'Sorry, there was an error processing your question. Please try again.';
                        chatBox.appendChild(errorDiv);
                    })
                    .finally(() => {
                        // Re-enable button
                        askButton.disabled = false;
                        askButton.textContent = 'Ask';
                    });
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content 