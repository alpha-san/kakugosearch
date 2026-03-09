# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**KakugoSearch** — a lightweight, AI-enhanced search engine in Rust. The binary is `kakugosearch`, served on port 7700 by default.

## Commands

```bash
# Build
cargo build --release

# Run (no AI, just full-text search)
cargo run

# Run with AI enabled
KAKUGOSEARCH_AI_API_KEY=sk-... cargo run

# Run tests
cargo test

# Run a single test
cargo test <test_name>
```

## Testing Policy

- Always run `cargo build` to confirm the project compiles before considering a change done.
- Always run `cargo test` after changing functionality to confirm nothing is broken.
- Always add tests when adding new functionality.

## Source File Layout

```
src/
├── main.rs
├── config.rs
├── ai/
│   ├── mod.rs          # build_provider() factory
│   ├── provider.rs     # AiProvider trait
│   ├── openai_compatible.rs
│   └── reranker.rs     # AiReranker (score blending)
├── api/
│   ├── mod.rs
│   ├── state.rs        # AppState, shared across handlers
│   └── routes.rs       # axum router + all handlers
├── index/
│   ├── mod.rs
│   └── search_index.rs # Tantivy wrapper
└── vector/
    ├── mod.rs
    └── store.rs        # In-process vector store
```

## Architecture

Search is a two-phase pipeline:

1. **BM25 full-text search** — `SearchIndex` wraps Tantivy. Each named index is persisted in `./data/<index_name>/`. Indexes are created on first use and held in a `DashMap` on `AppState`.

2. **AI reranking (optional)** — `AiReranker` takes the BM25 candidates and combines their scores with AI-derived cosine similarity scores. The blend is controlled by `ai_weight` (0.0 = pure BM25, 1.0 = pure AI). Falls back to BM25 silently on error.

`VectorStore` holds embeddings in-process as a brute-force cosine similarity store (serialized to `./data/vectors.json`). It is designed to be swapped for HNSW (usearch) or an external vector DB without changing the API.

All AI providers implement the `AiProvider` trait (`provider.rs`). Every provider in the codebase goes through `OpenAiCompatibleProvider` — OpenAI, Ollama, OpenRouter, vLLM, and LM Studio all speak the same `/v1/embeddings` API. The factory `ai::build_provider()` in `mod.rs` selects the right base URL per provider name.

## Adding a New AI Provider

1. Create `src/ai/<name>.rs` implementing `AiProvider` from `provider.rs`.
2. Add a match arm in `mod.rs` → `build_provider()`.
3. Add any config defaults in `config.rs` → `AiConfig`.

## Configuration

Config is loaded from `kakugosearch.toml` (falls back to defaults if missing). The only required value for AI is the API key, which can also be supplied via `KAKUGOSEARCH_AI_API_KEY`.

Key config sections: `[server]`, `[index]`, `[ai]`. See `kakugosearch.toml` for the full reference.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/indexes` | List indexes |
| POST | `/indexes/{name}/documents` | Add documents |
| GET | `/indexes/{name}/documents/{id}` | Fetch document |
| DELETE | `/indexes/{name}/documents/{id}` | Delete document |
| GET/POST | `/indexes/{name}/search` | Search (GET uses query params, POST uses JSON body) |

Search params: `q`, `limit` (default 20), `ai` (bool, default false), `ai_weight` (float 0–1, default 0.3).
