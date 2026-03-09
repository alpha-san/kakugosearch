use anyhow::Result;
use dashmap::DashMap;
use std::sync::Arc;

use crate::ai::{self, AiReranker};
use crate::config::AppConfig;
use crate::index::SearchIndex;
use crate::vector::VectorStore;

/// Shared state accessible by all request handlers.
#[derive(Clone)]
pub struct AppState {
    pub config: AppConfig,
    /// Map of index_name -> SearchIndex
    pub indexes: Arc<DashMap<String, Arc<SearchIndex>>>,
    /// AI reranker (None if AI is disabled)
    pub reranker: Option<Arc<AiReranker>>,
    /// Vector store for semantic search
    pub vector_store: Option<Arc<VectorStore>>,
}

impl AppState {
    pub async fn new(config: &AppConfig) -> Result<Self> {
        let indexes: Arc<DashMap<String, Arc<SearchIndex>>> = Arc::new(DashMap::new());

        // Create default index
        let default_index = SearchIndex::open_or_create(&config.index.data_dir, "default")?;
        indexes.insert("default".to_string(), Arc::new(default_index));

        // Initialize AI components if enabled
        let (reranker, vector_store) = if config.ai.enabled {
            let provider = ai::build_provider(&config.ai)?;
            let reranker = Arc::new(AiReranker::new(provider));

            let vs = VectorStore::new(
                config.ai.embedding_dims,
                Some(&config.index.data_dir),
            )?;

            (Some(reranker), Some(Arc::new(vs)))
        } else {
            (None, None)
        };

        Ok(Self {
            config: config.clone(),
            indexes,
            reranker,
            vector_store,
        })
    }

    /// Get or create a named index.
    pub fn get_or_create_index(&self, name: &str) -> Result<Arc<SearchIndex>> {
        if let Some(idx) = self.indexes.get(name) {
            return Ok(idx.clone());
        }

        let index = SearchIndex::open_or_create(&self.config.index.data_dir, name)?;
        let index = Arc::new(index);
        self.indexes.insert(name.to_string(), index.clone());
        Ok(index)
    }
}
