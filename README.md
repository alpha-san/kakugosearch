# KakugoSearch

A lightweight, AI-enhanced search engine with a pluggable provider architecture.

## Quick Start

```bash
# Clone and build
cargo build --release

# Run with defaults (no AI, just full-text search)
./target/release/kakugosearch

# Or with AI enabled (set your API key)
KAKUGOSEARCH_AI_API_KEY=sk-... cargo run
```

## API Usage

### Add Documents

```bash
curl -X POST http://localhost:7700/indexes/default/documents \
  -H "Content-Type: application/json" \
  -d '[
    {
      "id": "1",
      "title": "Introduction to Rust",
      "body": "Rust is a systems programming language focused on safety and performance.",
      "url": "https://rust-lang.org"
    },
    {
      "id": "2",
      "title": "Python for Data Science",
      "body": "Python is widely used in data science and machine learning applications.",
      "url": "https://python.org"
    }
  ]'
```

### Search (text only)

```bash
curl "http://localhost:7700/indexes/default/search?q=programming+language"
```

### Search (with AI reranking)

```bash
curl "http://localhost:7700/indexes/default/search?q=programming+language&ai=true&ai_weight=0.3"
```

### Search via POST (for complex queries)

```bash
curl -X POST http://localhost:7700/indexes/default/search \
  -H "Content-Type: application/json" \
  -d '{
    "q": "best language for building web servers",
    "limit": 10,
    "ai": true,
    "ai_weight": 0.5
  }'
```

### Delete a Document

```bash
curl -X DELETE http://localhost:7700/indexes/default/documents/1
```

## Configuration

Edit `kakugosearch.toml` or set environment variables:

| Setting | Env Var | Default |
|---------|---------|---------|
| AI API Key | `KAKUGOSEARCH_AI_API_KEY` | — |
| Server port | — | 7700 |

### AI Provider Examples

**OpenAI Direct:**
```toml
[ai]
enabled = true
provider = "openai"
embedding_model = "text-embedding-3-small"
embedding_dims = 768
```

**Ollama (local, free):**
```toml
[ai]
enabled = true
provider = "ollama"
embedding_model = "nomic-embed-text"
embedding_dims = 768
```

**OpenRouter (multi-model):**
```toml
[ai]
enabled = true
provider = "openrouter"
embedding_model = "openai/text-embedding-3-small"
embedding_dims = 768
```

**vLLM / LM Studio (self-hosted):**
```toml
[ai]
enabled = true
provider = "vllm"
base_url = "http://your-gpu-server:8000/v1"
embedding_model = "BAAI/bge-base-en-v1.5"
embedding_dims = 768
```

## Performance

Benchmarked on Apple M1 Pro. Run `cargo bench` to reproduce, or `cargo run --example load_test --release` for the concurrency results.

### Indexing throughput

| Batch size | Time per commit | Throughput |
|------------|----------------|------------|
| 100 docs   | 224 ms         | 447 docs/s |
| 1 000 docs | 223 ms         | 4 500 docs/s |
| 10 000 docs | 214 ms        | 46 800 docs/s |

Commit time is dominated by Tantivy's segment merge and fsync, not document count — hence throughput scales nearly linearly with batch size.

### Search latency (10 000-doc index, BM25)

| Query | p50 |
|-------|-----|
| `machine learning` | 92 µs |
| `memory safety concurrency` | 141 µs |
| `natural language processing algorithms` | 164 µs |
| `database query optimization` | 115 µs |
| `distributed systems performance` | 220 µs |

Latency scales with the number of posting list entries visited, not total index size.

### Search latency vs index size

```
Index size      p50 latency
──────────────────────────────
  1 000 docs     91 µs  ████░░░░░░░░░░░░░░░░
  5 000 docs    114 µs  █████░░░░░░░░░░░░░░░
 10 000 docs    143 µs  ██████░░░░░░░░░░░░░░
```

Sub-linear growth — query parsing and scorer initialisation are the dominant fixed costs at these index sizes.

## Architecture

```
┌─────────────────────────────────────────────┐
│                 HTTP API (axum)              │
│  POST /indexes/{name}/documents             │
│  GET  /indexes/{name}/search?q=...&ai=true  │
└────────┬────────────────────┬───────────────┘
         │                    │
    ┌────▼────┐         ┌────▼─────┐
    │ Tantivy │         │    AI    │
    │  (BM25) │         │ Reranker │
    └────┬────┘         └────┬─────┘
         │                   │
         │            ┌──────▼──────┐
         │            │  Provider   │
         │            │  Interface  │
         │            └──────┬──────┘
         │                   │
         │         ┌─────────┼─────────┐
         │         │         │         │
         │      OpenAI   Ollama   OpenRouter
         │      (+ any OpenAI-compatible API)
         │
    ┌────▼────┐
    │ Vector  │
    │  Store  │  (in-process, swappable)
    └─────────┘
```

## Adding a New AI Provider

1. Create `src/ai/my_provider.rs`
2. Implement the `AiProvider` trait
3. Add a match arm in `src/ai/mod.rs` → `build_provider()`
4. Add config defaults in `src/config.rs`

The `AiProvider` trait requires just 4 methods:
- `name()` — provider identifier
- `embed(text)` — single text → vector
- `rerank(query, docs)` — rerank candidates
- `embedding_dims()` — vector dimensions

## License

MIT
