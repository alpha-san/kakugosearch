use anyhow::Result;
use std::sync::Arc;

use super::provider::{AiProvider, ScoredDocument};

/// Orchestrates AI-enhanced search: takes full-text results and optionally
/// reranks them using the configured AI provider.
pub struct AiReranker {
    provider: Arc<dyn AiProvider>,
}

impl AiReranker {
    pub fn new(provider: Arc<dyn AiProvider>) -> Self {
        Self { provider }
    }

    /// Rerank a set of search results using the AI provider.
    ///
    /// `candidates` is a vec of (doc_id, doc_text, bm25_score) from the full-text search.
    /// Returns reranked results combining text relevance and semantic similarity.
    pub async fn rerank(
        &self,
        query: &str,
        candidates: Vec<(String, String, f32)>,
        ai_weight: f32, // 0.0 = pure text search, 1.0 = pure AI
    ) -> Result<Vec<ScoredDocument>> {
        let ai_weight = ai_weight.clamp(0.0, 1.0);
        let text_weight = 1.0 - ai_weight;

        if candidates.is_empty() {
            return Ok(vec![]);
        }

        // Normalize BM25 scores to 0-1 range
        let max_bm25 = candidates
            .iter()
            .map(|(_, _, s)| *s)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_bm25 = candidates
            .iter()
            .map(|(_, _, s)| *s)
            .fold(f32::INFINITY, f32::min);
        let bm25_range = (max_bm25 - min_bm25).max(f32::EPSILON);

        // Get AI reranking scores
        let doc_pairs: Vec<(String, String)> = candidates
            .iter()
            .map(|(id, text, _)| (id.clone(), text.clone()))
            .collect();

        let ai_scores = self.provider.rerank(query, doc_pairs).await?;

        // Build a map of doc_id -> ai_score
        let ai_score_map: std::collections::HashMap<String, f32> = ai_scores
            .into_iter()
            .map(|sd| (sd.id, sd.score))
            .collect();

        // Combine scores
        let mut combined: Vec<ScoredDocument> = candidates
            .iter()
            .map(|(id, _, bm25)| {
                let norm_bm25 = (bm25 - min_bm25) / bm25_range;
                let ai_score = ai_score_map.get(id).copied().unwrap_or(0.0);

                // Normalize AI score (cosine similarity is already -1 to 1, shift to 0-1)
                let norm_ai = (ai_score + 1.0) / 2.0;

                let combined_score = (text_weight * norm_bm25) + (ai_weight * norm_ai);

                ScoredDocument {
                    id: id.clone(),
                    score: combined_score,
                }
            })
            .collect();

        combined.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(combined)
    }

    /// Generate embedding for a document (for storage in vector index).
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.provider.embed(text).await
    }

    /// Batch embed documents.
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.provider.embed_batch(texts).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::provider::{AiProvider, ScoredDocument};
    use async_trait::async_trait;
    use std::sync::Arc;

    struct MockProvider {
        fixed_score: f32,
    }

    #[async_trait]
    impl AiProvider for MockProvider {
        fn name(&self) -> &str { "mock" }

        async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![1.0])
        }

        async fn rerank(
            &self,
            _query: &str,
            documents: Vec<(String, String)>,
        ) -> Result<Vec<ScoredDocument>> {
            Ok(documents
                .iter()
                .map(|(id, _)| ScoredDocument { id: id.clone(), score: self.fixed_score })
                .collect())
        }

        fn embedding_dims(&self) -> usize { 1 }
    }

    #[tokio::test]
    async fn test_rerank_empty_candidates() {
        let reranker = AiReranker::new(Arc::new(MockProvider { fixed_score: 0.5 }));
        let result = reranker.rerank("query", vec![], 0.5).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_rerank_pure_text_weight_preserves_bm25_order() {
        let reranker = AiReranker::new(Arc::new(MockProvider { fixed_score: 0.0 }));

        let candidates = vec![
            ("doc1".to_string(), "text".to_string(), 10.0_f32),
            ("doc2".to_string(), "text".to_string(), 5.0_f32),
            ("doc3".to_string(), "text".to_string(), 1.0_f32),
        ];

        // ai_weight=0.0 means ranking is purely by BM25 score
        let result = reranker.rerank("query", candidates, 0.0).await.unwrap();
        assert_eq!(result[0].id, "doc1");
        assert_eq!(result[1].id, "doc2");
        assert_eq!(result[2].id, "doc3");
    }

    #[tokio::test]
    async fn test_rerank_scores_are_normalized() {
        let reranker = AiReranker::new(Arc::new(MockProvider { fixed_score: 0.8 }));

        let candidates = vec![
            ("doc1".to_string(), "text".to_string(), 3.0_f32),
            ("doc2".to_string(), "text".to_string(), 1.0_f32),
        ];

        let result = reranker.rerank("query", candidates, 0.5).await.unwrap();
        for doc in &result {
            assert!(doc.score >= 0.0 && doc.score <= 1.0, "score {} out of [0, 1]", doc.score);
        }
    }
}
