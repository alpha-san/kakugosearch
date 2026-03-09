use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::provider::{AiProvider, ScoredDocument};

pub struct OpenAiCompatibleProvider {
    client: Client,
    base_url: String,
    api_key: String,
    embedding_model: String,
    rerank_model: Option<String>,
    dims: usize,
}

// ---- Request/Response types matching OpenAI embedding API ----

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    #[allow(dead_code)]
    index: usize,
}

impl OpenAiCompatibleProvider {
    pub fn new(
        base_url: String,
        api_key: String,
        embedding_model: String,
        rerank_model: Option<String>,
        dims: usize,
    ) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            embedding_model,
            rerank_model,
            dims,
        }
    }

    async fn call_embeddings(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url);

        let request = EmbeddingRequest {
            model: self.embedding_model.clone(),
            input: texts,
        };

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await
            .context("Failed to call embedding endpoint")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Embedding API returned {}: {}", status, body);
        }

        let resp: EmbeddingResponse = response.json().await?;

        // Sort by index to maintain input order
        let mut data = resp.data;
        data.sort_by_key(|d| d.index);

        Ok(data.into_iter().map(|d| d.embedding).collect())
    }
}

#[async_trait]
impl AiProvider for OpenAiCompatibleProvider {
    fn name(&self) -> &str {
        "openai_compatible"
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.call_embeddings(vec![text.to_string()]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Empty embedding response"))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // OpenAI supports batching natively — send all at once.
        // For very large batches, chunk into groups of 100.
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(100) {
            let results = self.call_embeddings(chunk.to_vec()).await?;
            all_embeddings.extend(results);
        }

        Ok(all_embeddings)
    }

    async fn rerank(
        &self,
        query: &str,
        documents: Vec<(String, String)>,
    ) -> Result<Vec<ScoredDocument>> {
        if self.rerank_model.is_none() {
            // Fallback: use cosine similarity between query embedding and doc embeddings
            return self.embedding_based_rerank(query, documents).await;
        }

        // TODO: Implement dedicated rerank API call for providers that support it
        // (Cohere, Jina, etc. have /rerank endpoints)
        self.embedding_based_rerank(query, documents).await
    }

    fn supports_reranking(&self) -> bool {
        true // We always support it via embedding fallback
    }

    fn embedding_dims(&self) -> usize {
        self.dims
    }
}

impl OpenAiCompatibleProvider {
    /// Rerank by computing cosine similarity between query and document embeddings.
    /// Works with any provider that supports embeddings.
    async fn embedding_based_rerank(
        &self,
        query: &str,
        documents: Vec<(String, String)>,
    ) -> Result<Vec<ScoredDocument>> {
        let query_embedding = self.embed(query).await?;

        let doc_texts: Vec<String> = documents.iter().map(|(_, text)| text.clone()).collect();
        let doc_embeddings = self.embed_batch(&doc_texts).await?;

        let mut scored: Vec<ScoredDocument> = documents
            .iter()
            .zip(doc_embeddings.iter())
            .map(|((id, _), doc_emb)| ScoredDocument {
                id: id.clone(),
                score: cosine_similarity(&query_embedding, doc_emb),
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}
