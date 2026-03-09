use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

/// In-process vector store for semantic search.
///
/// Uses a simple brute-force approach for the POC.
/// For production with millions of records, swap this implementation
/// for usearch HNSW or an external vector DB — the interface stays the same.
///
/// The trait-based design means you can later implement:
/// - UsearchVectorStore (in-process HNSW)
/// - QdrantVectorStore (external)
/// - PineconeVectorStore (managed)
pub struct VectorStore {
    dims: usize,
    /// doc_id -> embedding vector
    vectors: RwLock<HashMap<String, Vec<f32>>>,
    /// Path for persistence
    data_path: Option<std::path::PathBuf>,
}

#[derive(Debug)]
pub struct VectorSearchResult {
    pub id: String,
    pub score: f32,
}

impl VectorStore {
    pub fn new(dims: usize, data_dir: Option<impl AsRef<Path>>) -> Result<Self> {
        let data_path = data_dir.map(|p| p.as_ref().join("vectors.json"));

        let vectors = if let Some(ref path) = data_path {
            if path.exists() {
                let data = std::fs::read_to_string(path)
                    .context("Failed to read vector store")?;
                serde_json::from_str(&data)
                    .context("Failed to parse vector store")?
            } else {
                HashMap::new()
            }
        } else {
            HashMap::new()
        };

        Ok(Self {
            dims,
            vectors: RwLock::new(vectors),
            data_path,
        })
    }

    /// Insert or update a vector for a document.
    pub fn upsert(&self, id: &str, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dims {
            anyhow::bail!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dims,
                vector.len()
            );
        }

        let mut store = self.vectors.write().map_err(|e| anyhow::anyhow!("Lock: {}", e))?;
        store.insert(id.to_string(), vector);
        Ok(())
    }

    /// Batch insert vectors.
    pub fn upsert_batch(&self, entries: Vec<(String, Vec<f32>)>) -> Result<usize> {
        let mut store = self.vectors.write().map_err(|e| anyhow::anyhow!("Lock: {}", e))?;
        let mut count = 0;
        for (id, vector) in entries {
            if vector.len() != self.dims {
                tracing::warn!("Skipping doc {} — wrong dims ({} != {})", id, vector.len(), self.dims);
                continue;
            }
            store.insert(id, vector);
            count += 1;
        }
        Ok(count)
    }

    /// Delete a vector by document ID.
    pub fn delete(&self, id: &str) -> Result<bool> {
        let mut store = self.vectors.write().map_err(|e| anyhow::anyhow!("Lock: {}", e))?;
        Ok(store.remove(id).is_some())
    }

    /// Find the k nearest neighbors to the given query vector.
    /// Uses brute-force cosine similarity (fine for POC, swap for HNSW in prod).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<VectorSearchResult>> {
        if query.len() != self.dims {
            anyhow::bail!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.dims,
                query.len()
            );
        }

        let store = self.vectors.read().map_err(|e| anyhow::anyhow!("Lock: {}", e))?;

        let mut results: Vec<VectorSearchResult> = store
            .iter()
            .map(|(id, vec)| VectorSearchResult {
                id: id.clone(),
                score: cosine_similarity(query, vec),
            })
            .collect();

        // Sort descending by score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        Ok(results)
    }

    /// Persist vectors to disk.
    pub fn save(&self) -> Result<()> {
        if let Some(ref path) = self.data_path {
            let store = self.vectors.read().map_err(|e| anyhow::anyhow!("Lock: {}", e))?;
            let data = serde_json::to_string(&*store)?;
            std::fs::write(path, data)?;
            tracing::info!("Vector store saved ({} vectors)", store.len());
        }
        Ok(())
    }

    /// Number of stored vectors.
    pub fn len(&self) -> usize {
        self.vectors.read().map(|s| s.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
