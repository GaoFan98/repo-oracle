# Repo Oracle

Repo Oracle Hub is an AI-powered documentation assistant and codebase Q&A system for GitLab repositories. It automatically generates docstrings for Python code and provides a semantic search interface to ask questions about the codebase.

## Features

- **Webhook Listener**: Automatically processes GitLab merge requests when merged to `main`
- **AI Docstring Generation**: Uses Vertex AI Codey to add docstrings to Python functions and classes
- **Knowledge Indexing**: Creates a semantic search index of your codebase
- **Q&A Endpoint**: Answer questions about your code with AI-generated responses

## Prerequisites

- Docker and Docker Compose
- A GitLab account with a project to integrate with
- Google Cloud Platform account with Vertex AI API access
- A service account with permissions to use Vertex AI

## Setup Instructions

### 1. Create a `.env` file with the following variables:

```
GITLAB_TOKEN=your_gitlab_personal_access_token
GITLAB_HOST=https://gitlab.com  # Or your self-hosted GitLab URL
WEBHOOK_SECRET=your_webhook_secret
PROJECT_ID=your_gitlab_project_id
REPO_CLONE_URL=https://oauth2:{token}@gitlab.com/your-username/your-project.git
GOOGLE_PROJECT=your_google_cloud_project_id
GOOGLE_LOCATION=us-central1  # Or your preferred region
```

### 2. Create a service account in Google Cloud

1. Navigate to the IAM & Admin > Service Accounts section in the Google Cloud Console
2. Create a new service account with the "Vertex AI User" role
3. Create a key for this service account and download it as JSON
4. Save the JSON key file as `sa.json` in the project root directory

### 3. Start the application

```bash
docker-compose up -d
```

### 4. Configure the GitLab webhook

1. Go to your GitLab project
2. Navigate to Settings > Webhooks
3. Add a new webhook with:
   - URL: `http://your-server:8000/webhook`
   - Secret token: The same value as `WEBHOOK_SECRET` in your `.env` file
   - Events: Select "Merge requests"
4. Click "Add webhook"

## Usage

### Reindex the Repository

To manually trigger a reindex of the repository:

```bash
curl -X POST http://localhost:8000/reindex
```

### Ask Questions About the Codebase

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Where is token validation handled?"}'
```

## How It Works

1. When a merge request is merged to `main`, the webhook is triggered
2. The app retrieves the changes and identifies Python files
3. For each modified Python file, it parses the code to find functions/classes without docstrings
4. It uses Vertex AI Codey to generate appropriate docstrings
5. The app creates a new branch, commits the changes, and opens a merge request
6. The codebase is indexed to enable semantic search
7. Developers can ask questions through the `/ask` endpoint to get context-aware answers

## Docker Volumes

The application uses a Docker volume to persist the FAISS vector index:

- `./data:/app/data` - Stores the vector index and metadata

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 