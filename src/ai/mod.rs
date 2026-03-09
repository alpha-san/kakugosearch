pub mod provider;
pub mod openai_compatible;
pub mod reranker;

pub use provider::{AiProvider, EmbeddingProvider, RerankProvider, ScoredDocument};
pub use openai_compatible::OpenAiCompatibleProvider;
pub use reranker::AiReranker;

use crate::config::AiConfig;
use anyhow::Result;
use std::sync::Arc;

/// Factory: build the right provider from config.
/// This is where the "plug and play" happens — add new match arms for new providers.
pub fn build_provider(config: &AiConfig) -> Result<Arc<dyn AiProvider>> {
    let api_key = config
        .api_key
        .clone()
        .or_else(|| std::env::var("KAKUGOSEARCH_AI_API_KEY").ok())
        .unwrap_or_default();

    match config.provider.as_str() {
        "openai_compatible" | "openai" | "openrouter" | "ollama" | "vllm" | "lmstudio" => {
            let base_url = config.base_url.clone().unwrap_or_else(|| {
                match config.provider.as_str() {
                    "openrouter" => "https://openrouter.ai/api/v1".to_string(),
                    "ollama" => "http://localhost:11434/v1".to_string(),
                    "lmstudio" => "http://localhost:1234/v1".to_string(),
                    "vllm" => "http://localhost:8000/v1".to_string(),
                    _ => "https://api.openai.com/v1".to_string(),
                }
            });

            Ok(Arc::new(OpenAiCompatibleProvider::new(
                base_url,
                api_key,
                config.embedding_model.clone(),
                config.rerank_model.clone(),
                config.embedding_dims,
            )))
        }

        // -------------------------------------------------------
        // ADD NEW PROVIDERS HERE:
        //
        // "cohere" => Ok(Arc::new(CohereProvider::new(...))),
        // "anthropic" => Ok(Arc::new(AnthropicProvider::new(...))),
        // "custom" => Ok(Arc::new(CustomHttpProvider::new(...))),
        // -------------------------------------------------------

        other => anyhow::bail!(
            "Unknown AI provider '{}'. Supported: openai_compatible, openrouter, ollama, vllm, lmstudio",
            other
        ),
    }
}
