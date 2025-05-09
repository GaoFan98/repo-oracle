import os
import sys
import uvicorn
import argparse

from .index import reindex_repository

def main():
    """Command-line entry point for Repo Oracle."""
    parser = argparse.ArgumentParser(description="Repo Oracle CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index the repository")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the web server")
    serve_parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to bind to"
    )
    serve_parser.add_argument(
        "--port", 
        type=int, 
        default=int(os.environ.get("PORT", 8000)), 
        help="Port to bind to"
    )
    
    args = parser.parse_args()
    
    if args.command == "index":
        print("Indexing repository...")
        reindex_repository()
        print("Indexing complete")
    elif args.command == "serve":
        print(f"Starting server on {args.host}:{args.port}")
        uvicorn.run(
            "devknowledge.main:app", 
            host=args.host, 
            port=args.port,
            reload=False
        )
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 