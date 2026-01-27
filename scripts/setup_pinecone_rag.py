#!/usr/bin/env python3
"""
AstroLlama - Pinecone RAG Setup
Setup and manage the Pinecone vector database for RAG.

Usage:
    # Setup Pinecone index
    python setup_pinecone_rag.py setup
    
    # Ingest documents
    python setup_pinecone_rag.py ingest --source ./data/rag/
    
    # Test retrieval
    python setup_pinecone_rag.py test --query "How do I query Gaia DR3?"
    
    # Clear index
    python setup_pinecone_rag.py clear
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Check for required packages
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("Warning: pinecone-client not installed. Run: pip install pinecone-client")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Run: pip install sentence-transformers")


# =============================================================================
# Configuration
# =============================================================================

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX", "astrollama-knowledge")
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSION = 1024
CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 50


# =============================================================================
# Embedding Model
# =============================================================================

_embedding_model = None

def get_embedding_model():
    """Get or create the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("sentence-transformers not installed")
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts."""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()


# =============================================================================
# Pinecone Operations
# =============================================================================

def get_pinecone_client():
    """Get Pinecone client."""
    if not PINECONE_AVAILABLE:
        raise ImportError("pinecone-client not installed")
    
    # First try environment variable
    api_key = PINECONE_API_KEY
    
    if not api_key:
        # Try to get from AWS Secrets Manager
        try:
            import boto3
            client = boto3.client("secretsmanager", region_name="us-west-2")
            response = client.get_secret_value(SecretId="astrollama/api-keys")
            secrets = json.loads(response["SecretString"])
            api_key = secrets.get("PINECONE_API_KEY")
        except Exception as e:
            print(f"Note: Could not get from Secrets Manager: {e}")
    
    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY not set. Either:\n"
            "  1. Set environment variable: export PINECONE_API_KEY=your-key\n"
            "  2. Add to AWS Secrets Manager: astrollama/api-keys"
        )
    
    return Pinecone(api_key=api_key)


def setup_index():
    """Create Pinecone index if it doesn't exist."""
    pc = get_pinecone_client()
    
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME in existing_indexes:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists")
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        print(f"  Total vectors: {stats.total_vector_count}")
        return index
    
    print(f"Creating index '{PINECONE_INDEX_NAME}'...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-west-2"  # Match your Bedrock region
        )
    )
    
    print(f"Index '{PINECONE_INDEX_NAME}' created successfully!")
    return pc.Index(PINECONE_INDEX_NAME)


def get_index():
    """Get existing Pinecone index."""
    pc = get_pinecone_client()
    return pc.Index(PINECONE_INDEX_NAME)


# =============================================================================
# Document Processing
# =============================================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    # Simple word-based chunking
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) < chunk_size // 4:  # Skip very small chunks
            continue
        chunks.append(" ".join(chunk_words))
    
    return chunks


def process_file(filepath: Path) -> List[Dict[str, Any]]:
    """Process a single file into chunks with metadata."""
    chunks = []
    
    suffix = filepath.suffix.lower()
    
    try:
        if suffix in [".txt", ".md", ".rst"]:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        elif suffix == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                text = json.dumps(data, indent=2)
        elif suffix == ".pdf":
            # Try to extract text from PDF
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(filepath)
                text = ""
                for page in doc:
                    text += page.get_text()
            except ImportError:
                print(f"  Skipping PDF (pymupdf not installed): {filepath}")
                return []
        else:
            print(f"  Skipping unsupported file type: {filepath}")
            return []
        
        # Chunk the text
        text_chunks = chunk_text(text)
        
        for i, chunk in enumerate(text_chunks):
            chunks.append({
                "id": f"{filepath.stem}_{i}",
                "text": chunk,
                "metadata": {
                    "source": str(filepath.name),
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                }
            })
        
        print(f"  Processed {filepath.name}: {len(text_chunks)} chunks")
        
    except Exception as e:
        print(f"  Error processing {filepath}: {e}")
    
    return chunks


def ingest_documents(source_dir: str, batch_size: int = 100):
    """Ingest documents from a directory into Pinecone."""
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Source directory not found: {source_dir}")
        return
    
    # Get or create index
    index = setup_index()
    
    # Collect all documents
    all_chunks = []
    
    for filepath in source_path.rglob("*"):
        if filepath.is_file() and not filepath.name.startswith("."):
            chunks = process_file(filepath)
            all_chunks.extend(chunks)
    
    if not all_chunks:
        print("No documents to ingest")
        return
    
    print(f"\nTotal chunks to embed: {len(all_chunks)}")
    
    # Embed and upsert in batches
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        
        # Get embeddings
        texts = [c["text"] for c in batch]
        embeddings = embed_texts(texts)
        
        # Prepare vectors for Pinecone
        vectors = []
        for chunk, embedding in zip(batch, embeddings):
            vectors.append({
                "id": chunk["id"],
                "values": embedding,
                "metadata": {
                    **chunk["metadata"],
                    "text": chunk["text"][:1000]  # Store truncated text in metadata
                }
            })
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors)
        print(f"  Upserted batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")
    
    print(f"\nIngestion complete! Total vectors: {len(all_chunks)}")


# =============================================================================
# Retrieval
# =============================================================================

def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve relevant documents for a query."""
    index = get_index()
    
    # Embed query
    query_embedding = embed_texts([query])[0]
    
    # Search
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return results.matches


def test_retrieval(query: str, top_k: int = 5):
    """Test retrieval with a query."""
    print(f"\nQuery: {query}")
    print("-" * 60)
    
    results = retrieve(query, top_k)
    
    if not results:
        print("No results found")
        return
    
    for i, match in enumerate(results):
        print(f"\n[{i+1}] Score: {match.score:.4f}")
        print(f"    Source: {match.metadata.get('source', 'Unknown')}")
        print(f"    Text: {match.metadata.get('text', '')[:200]}...")


def clear_index():
    """Clear all vectors from the index."""
    index = get_index()
    
    # Delete all vectors
    index.delete(delete_all=True)
    print(f"Cleared all vectors from '{PINECONE_INDEX_NAME}'")


# =============================================================================
# RAG Chain
# =============================================================================

def create_rag_context(query: str, top_k: int = 5) -> str:
    """Create context string from retrieved documents."""
    results = retrieve(query, top_k)
    
    if not results:
        return ""
    
    context_parts = []
    for match in results:
        source = match.metadata.get("source", "Unknown")
        text = match.metadata.get("text", "")
        context_parts.append(f"[Source: {source}]\n{text}")
    
    return "\n\n---\n\n".join(context_parts)


def rag_prompt(query: str, context: str) -> str:
    """Create a RAG-augmented prompt."""
    return f"""Use the following context to answer the question. If the context doesn't contain relevant information, say so and answer based on your general knowledge.

Context:
{context}

Question: {query}

Answer:"""


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AstroLlama Pinecone RAG Setup")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Setup command
    subparsers.add_parser("setup", help="Setup Pinecone index")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--source", default="./data/rag/", help="Source directory")
    ingest_parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test retrieval")
    test_parser.add_argument("--query", required=True, help="Test query")
    test_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    
    # Clear command
    subparsers.add_parser("clear", help="Clear all vectors")
    
    # Stats command
    subparsers.add_parser("stats", help="Show index statistics")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        setup_index()
    
    elif args.command == "ingest":
        ingest_documents(args.source, args.batch_size)
    
    elif args.command == "test":
        test_retrieval(args.query, args.top_k)
    
    elif args.command == "clear":
        confirm = input("Are you sure you want to clear all vectors? (yes/no): ")
        if confirm.lower() == "yes":
            clear_index()
    
    elif args.command == "stats":
        index = get_index()
        stats = index.describe_index_stats()
        print(f"\nIndex: {PINECONE_INDEX_NAME}")
        print(f"Total vectors: {stats.total_vector_count}")
        print(f"Dimension: {stats.dimension}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
