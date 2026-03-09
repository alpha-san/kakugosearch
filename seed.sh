#!/usr/bin/env bash
# Seed LightSearch with sample documents for testing.
# Usage: ./examples/seed.sh

BASE_URL="${LIGHTSEARCH_URL:-http://localhost:7700}"
INDEX="default"

echo "Seeding $BASE_URL/indexes/$INDEX ..."

curl -s -X POST "$BASE_URL/indexes/$INDEX/documents" \
  -H "Content-Type: application/json" \
  -d '[
  {
    "id": "rust-intro",
    "title": "Introduction to Rust Programming",
    "body": "Rust is a multi-paradigm systems programming language focused on safety, especially safe concurrency. It is syntactically similar to C++ but provides memory safety without garbage collection. Rust achieves memory safety through a system of ownership with a set of rules that the compiler checks at compile time.",
    "url": "https://doc.rust-lang.org/book/",
    "metadata": {"tags": ["programming", "systems", "rust"]}
  },
  {
    "id": "python-ml",
    "title": "Python for Machine Learning",
    "body": "Python has become the dominant language for machine learning and data science. Libraries like scikit-learn, TensorFlow, and PyTorch provide powerful tools for building ML models. The language simplicity and extensive ecosystem make it ideal for rapid prototyping and research.",
    "url": "https://scikit-learn.org",
    "metadata": {"tags": ["programming", "ml", "python"]}
  },
  {
    "id": "search-engines",
    "title": "How Search Engines Work",
    "body": "Search engines use inverted indexes to quickly find documents containing specific terms. The process involves crawling web pages, tokenizing content, building an index, and ranking results using algorithms like BM25 or TF-IDF. Modern search engines also incorporate semantic understanding using neural embeddings.",
    "url": "https://en.wikipedia.org/wiki/Search_engine",
    "metadata": {"tags": ["search", "information-retrieval"]}
  },
  {
    "id": "vector-db",
    "title": "Vector Databases Explained",
    "body": "Vector databases store high-dimensional vectors and enable fast similarity search using algorithms like HNSW (Hierarchical Navigable Small World). They are essential for semantic search, recommendation systems, and RAG (Retrieval Augmented Generation) pipelines. Popular options include Qdrant, Pinecone, and Weaviate.",
    "metadata": {"tags": ["databases", "vectors", "ai"]}
  },
  {
    "id": "web-frameworks",
    "title": "Comparing Web Frameworks: Axum vs Actix vs Rocket",
    "body": "Rust web frameworks offer high performance with strong type safety. Axum leverages the tower ecosystem and provides an ergonomic API. Actix-web is known for raw performance benchmarks. Rocket focuses on developer experience with macros and code generation. All three are production-ready choices for building web services.",
    "metadata": {"tags": ["rust", "web", "frameworks"]}
  }
]' | python3 -m json.tool 2>/dev/null || cat

echo ""
echo "Done! Try searching:"
echo "  curl \"$BASE_URL/indexes/$INDEX/search?q=programming+language\""
echo "  curl \"$BASE_URL/indexes/$INDEX/search?q=how+do+search+engines+find+documents&ai=true\""
