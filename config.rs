use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub index: IndexConfig,
    pub ai: AiConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

#[derive(Debug, Deserialize, Clone)]
pub struct IndexConfig {
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
    /// Max number of documents returned per search
    #[serde(default = "default_max_results")]
    pub max_results: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AiConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_provider")]
    pub provider: String,
    /// Base URL for the AI provider (OpenAI-compatible endpoint)
    #[serde(default)]
    pub base_url: Option<String>,
    /// API key (can also be set via LIGHTSEARCH_AI_API_KEY env var)
    #[serde(default)]
    pub api_key: Option<String>,
    /// Model name for embeddings
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,
    /// Model name for reranking (if supported)
    #[serde(default)]
    pub rerank_model: Option<String>,
    /// Embedding dimensions
    #[serde(default = "default_embedding_dims")]
    pub embedding_dims: usize,
}

fn default_host() -> String { "0.0.0.0".to_string() }
fn default_port() -> u16 { 7700 }
fn default_data_dir() -> String { "./data".to_string() }
fn default_max_results() -> usize { 20 }
fn default_provider() -> String { "openai_compatible".to_string() }
fn default_embedding_model() -> String { "text-embedding-3-small".to_string() }
fn default_embedding_dims() -> usize { 768 }

impl AppConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        if path.exists() {
            let content = std::fs::read_to_string(path)?;
            let config: AppConfig = toml::from_str(&content)?;
            Ok(config)
        } else {
            tracing::warn!("Config file not found at {:?}, using defaults", path);
            Ok(Self::default())
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: default_host(),
                port: default_port(),
            },
            index: IndexConfig {
                data_dir: default_data_dir(),
                max_results: default_max_results(),
            },
            ai: AiConfig {
                enabled: false,
                provider: default_provider(),
                base_url: None,
                api_key: None,
                embedding_model: default_embedding_model(),
                rerank_model: None,
                embedding_dims: default_embedding_dims(),
            },
        }
    }
}
