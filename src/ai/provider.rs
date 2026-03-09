use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// A document with a relevance score from AI reranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredDocument {
    pub id: String,
    pub score: f32,
}

// ============================================================
// CORE TRAIT: This is the plugin contract.
// Implement this trait to add a new AI provider.
// ============================================================

/// Combined trait for providers that support both embedding and reranking.
/// If a provider only supports one, it can return an error for the other.
#[async_trait]
pub trait AiProvider: Send + Sync {
    /// Provider name (for logging/config)
    fn name(&self) -> &str;

    /// Generate embedding vector for a single text input.
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for a batch of texts.
    /// Default implementation calls embed() in a loop — override for efficiency.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// Rerank candidate documents given a query.
    /// Returns documents sorted by relevance (highest first).
    async fn rerank(
        &self,
        query: &str,
        documents: Vec<(String, String)>, // (doc_id, doc_text)
    ) -> Result<Vec<ScoredDocument>>;

    /// Check if this provider supports embeddings.
    fn supports_embeddings(&self) -> bool { true }

    /// Check if this provider supports reranking.
    fn supports_reranking(&self) -> bool { false }

    /// Embedding dimensions this provider produces.
    fn embedding_dims(&self) -> usize;
}

// Convenience sub-traits for when you only need one capability.

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dims(&self) -> usize;
}

#[async_trait]
pub trait RerankProvider: Send + Sync {
    async fn rerank(
        &self,
        query: &str,
        documents: Vec<(String, String)>,
    ) -> Result<Vec<ScoredDocument>>;
}
