use axum::{
    extract::{Json, Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use super::AppState;
use crate::index::search_index::Document;

// ============================================================
// Router
// ============================================================

pub fn router(state: AppState) -> Router {
    Router::new()
        // Health
        .route("/health", get(health))
        // Document management
        .route("/indexes/:index_name/documents", post(add_documents))
        .route("/indexes/:index_name/documents/:doc_id", get(get_document))
        .route("/indexes/:index_name/documents/:doc_id", delete(delete_document))
        // Search
        .route("/indexes/:index_name/search", get(search))
        .route("/indexes/:index_name/search", post(search_post))
        // Index management
        .route("/indexes", get(list_indexes))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

// ============================================================
// Request / Response types
// ============================================================

#[derive(Deserialize)]
struct SearchParams {
    q: String,
    #[serde(default = "default_limit")]
    limit: usize,
    /// Enable AI reranking for this query (default: false)
    #[serde(default)]
    ai: bool,
    /// AI weight: 0.0 = pure text, 1.0 = pure semantic (default: 0.3)
    #[serde(default = "default_ai_weight")]
    ai_weight: f32,
}

#[derive(Deserialize)]
struct SearchBody {
    q: String,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    ai: bool,
    #[serde(default = "default_ai_weight")]
    ai_weight: f32,
}

fn default_limit() -> usize { 20 }
fn default_ai_weight() -> f32 { 0.3 }

#[derive(Serialize)]
struct SearchResponse {
    hits: Vec<HitResponse>,
    query: String,
    processing_time_ms: u64,
    ai_enabled: bool,
}

#[derive(Serialize)]
struct HitResponse {
    id: String,
    title: String,
    body: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<String>,
    score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct AddDocumentsResponse {
    added: usize,
    index: String,
}

#[derive(Serialize)]
struct IndexInfo {
    name: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// ============================================================
// Handlers
// ============================================================

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

async fn list_indexes(State(state): State<AppState>) -> impl IntoResponse {
    let indexes: Vec<IndexInfo> = state
        .indexes
        .iter()
        .map(|entry| IndexInfo {
            name: entry.key().clone(),
        })
        .collect();

    Json(indexes)
}

async fn add_documents(
    State(state): State<AppState>,
    Path(index_name): Path<String>,
    Json(mut documents): Json<Vec<Document>>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    // Assign IDs to documents that don't have one
    for doc in &mut documents {
        if doc.id.is_empty() {
            doc.id = Uuid::new_v4().to_string();
        }
    }

    let index = state
        .get_or_create_index(&index_name)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: e.to_string() }),
            )
        })?;

    let count = index.add_documents(&documents).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;

    index.commit().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;

    // If AI is enabled, generate and store embeddings
    if let (Some(reranker), Some(vector_store)) = (&state.reranker, &state.vector_store) {
        let texts: Vec<String> = documents
            .iter()
            .map(|d| format!("{} {}", d.title, d.body))
            .collect();

        match reranker.embed_batch(&texts).await {
            Ok(embeddings) => {
                let entries: Vec<(String, Vec<f32>)> = documents
                    .iter()
                    .zip(embeddings.into_iter())
                    .map(|(doc, emb)| (doc.id.clone(), emb))
                    .collect();

                if let Err(e) = vector_store.upsert_batch(entries) {
                    tracing::warn!("Failed to store embeddings: {}", e);
                }
                if let Err(e) = vector_store.save() {
                    tracing::warn!("Failed to persist vector store: {}", e);
                }
            }
            Err(e) => {
                tracing::warn!("Failed to generate embeddings: {}", e);
            }
        }
    }

    Ok((
        StatusCode::ACCEPTED,
        Json(AddDocumentsResponse {
            added: count,
            index: index_name,
        }),
    ))
}

async fn get_document(
    State(state): State<AppState>,
    Path((index_name, doc_id)): Path<(String, String)>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let index = state.get_or_create_index(&index_name).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;

    match index.get_document(&doc_id) {
        Ok(Some(result)) => Ok(Json(HitResponse {
            id: result.id,
            title: result.title,
            body: result.body,
            url: result.url,
            score: result.score,
            metadata: result.metadata,
        })),
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Document '{}' not found", doc_id),
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e.to_string() }),
        )),
    }
}

async fn delete_document(
    State(state): State<AppState>,
    Path((index_name, doc_id)): Path<(String, String)>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let index = state.get_or_create_index(&index_name).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;

    index.delete_document(&doc_id).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;

    index.commit().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;

    // Also remove from vector store
    if let Some(vector_store) = &state.vector_store {
        let _ = vector_store.delete(&doc_id);
    }

    Ok(StatusCode::NO_CONTENT)
}

/// Search via GET query params
async fn search(
    State(state): State<AppState>,
    Path(index_name): Path<String>,
    Query(params): Query<SearchParams>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    do_search(state, index_name, params.q, params.limit, params.ai, params.ai_weight).await
}

/// Search via POST body (for complex queries)
async fn search_post(
    State(state): State<AppState>,
    Path(index_name): Path<String>,
    Json(body): Json<SearchBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    do_search(state, index_name, body.q, body.limit, body.ai, body.ai_weight).await
}

async fn do_search(
    state: AppState,
    index_name: String,
    query: String,
    limit: usize,
    ai_enabled: bool,
    ai_weight: f32,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();

    let index = state.get_or_create_index(&index_name).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;

    // Phase 1: Full-text search (BM25)
    let text_results = index.search(&query, limit * 2).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;

    // Phase 2: Optionally rerank with AI
    let hits = if ai_enabled {
        if let Some(reranker) = &state.reranker {
            let candidates: Vec<(String, String, f32)> = text_results
                .iter()
                .map(|r| (r.id.clone(), format!("{} {}", r.title, r.body), r.score))
                .collect();

            match reranker.rerank(&query, candidates, ai_weight).await {
                Ok(reranked) => {
                    // Map reranked scores back to full results
                    let result_map: std::collections::HashMap<String, &crate::index::search_index::SearchResult> =
                        text_results.iter().map(|r| (r.id.clone(), r)).collect();

                    reranked
                        .into_iter()
                        .take(limit)
                        .filter_map(|scored| {
                            result_map.get(&scored.id).map(|r| HitResponse {
                                id: r.id.clone(),
                                title: r.title.clone(),
                                body: r.body.clone(),
                                url: r.url.clone(),
                                score: scored.score,
                                metadata: r.metadata.clone(),
                            })
                        })
                        .collect()
                }
                Err(e) => {
                    tracing::warn!("AI reranking failed, falling back to text search: {}", e);
                    text_results_to_hits(text_results, limit)
                }
            }
        } else {
            text_results_to_hits(text_results, limit)
        }
    } else {
        text_results_to_hits(text_results, limit)
    };

    let elapsed = start.elapsed();

    Ok(Json(SearchResponse {
        hits,
        query,
        processing_time_ms: elapsed.as_millis() as u64,
        ai_enabled: ai_enabled && state.reranker.is_some(),
    }))
}

fn text_results_to_hits(
    results: Vec<crate::index::search_index::SearchResult>,
    limit: usize,
) -> Vec<HitResponse> {
    results
        .into_iter()
        .take(limit)
        .map(|r| HitResponse {
            id: r.id,
            title: r.title,
            body: r.body,
            url: r.url,
            score: r.score,
            metadata: r.metadata,
        })
        .collect()
}
