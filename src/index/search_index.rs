use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{TantivyDocument, schema::OwnedValue};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy};

/// A document stored in the search index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique document ID (auto-generated if not provided)
    pub id: String,
    /// Primary searchable text content
    pub title: String,
    /// Full body text
    pub body: String,
    /// Optional URL or source reference
    #[serde(default)]
    pub url: Option<String>,
    /// Arbitrary metadata as JSON
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

/// Search result with score.
#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub id: String,
    pub title: String,
    pub body: String,
    pub url: Option<String>,
    pub score: f32,
    pub metadata: Option<serde_json::Value>,
}

/// Wraps a Tantivy index for full-text search.
pub struct SearchIndex {
    index: Index,
    reader: IndexReader,
    writer: std::sync::Mutex<IndexWriter>,
    // Schema fields
    f_id: Field,
    f_title: Field,
    f_body: Field,
    f_url: Field,
    f_metadata: Field,
}

impl SearchIndex {
    /// Create or open an index at the given directory.
    pub fn open_or_create(data_dir: impl AsRef<Path>, index_name: &str) -> Result<Self> {
        let index_path = data_dir.as_ref().join(index_name);
        std::fs::create_dir_all(&index_path)?;

        let mut schema_builder = Schema::builder();

        let f_id = schema_builder.add_text_field("id", STRING | STORED);
        let f_title = schema_builder.add_text_field("title", TEXT | STORED);
        let f_body = schema_builder.add_text_field("body", TEXT | STORED);
        let f_url = schema_builder.add_text_field("url", STORED);
        let f_metadata = schema_builder.add_text_field("metadata", STORED);

        let schema = schema_builder.build();

        let index = if index_path.join("meta.json").exists() {
            Index::open_in_dir(&index_path)
                .context("Failed to open existing index")?
        } else {
            Index::create_in_dir(&index_path, schema.clone())
                .context("Failed to create new index")?
        };

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;

        let writer = index.writer(50_000_000)?; // 50MB heap

        Ok(Self {
            index,
            reader,
            writer: std::sync::Mutex::new(writer),
            f_id,
            f_title,
            f_body,
            f_url,
            f_metadata,
        })
    }

    /// Add a single document to the index.
    pub fn add_document(&self, document: &Document) -> Result<()> {
        let writer = self.writer.lock().map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;

        let metadata_str = document
            .metadata
            .as_ref()
            .map(|m| serde_json::to_string(m).unwrap_or_default())
            .unwrap_or_default();

        writer.add_document(doc!(
            self.f_id => document.id.as_str(),
            self.f_title => document.title.as_str(),
            self.f_body => document.body.as_str(),
            self.f_url => document.url.as_deref().unwrap_or(""),
            self.f_metadata => metadata_str.as_str(),
        ))?;

        Ok(())
    }

    /// Add multiple documents in a batch.
    pub fn add_documents(&self, documents: &[Document]) -> Result<usize> {
        let writer = self.writer.lock().map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;

        let mut count = 0;
        for document in documents {
            let metadata_str = document
                .metadata
                .as_ref()
                .map(|m| serde_json::to_string(m).unwrap_or_default())
                .unwrap_or_default();

            writer.add_document(doc!(
                self.f_id => document.id.as_str(),
                self.f_title => document.title.as_str(),
                self.f_body => document.body.as_str(),
                self.f_url => document.url.as_deref().unwrap_or(""),
                self.f_metadata => metadata_str.as_str(),
            ))?;
            count += 1;
        }

        Ok(count)
    }

    /// Commit pending writes to disk and reload the reader so changes are immediately visible.
    pub fn commit(&self) -> Result<()> {
        let mut writer = self.writer.lock().map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        writer.commit()?;
        self.reader.reload()?;
        Ok(())
    }

    /// Delete a document by ID.
    pub fn delete_document(&self, id: &str) -> Result<()> {
        let writer = self.writer.lock().map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        let term = tantivy::Term::from_field_text(self.f_id, id);
        writer.delete_term(term);
        Ok(())
    }

    /// Full-text search. Returns results sorted by BM25 relevance.
    pub fn search(&self, query_str: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.f_title, self.f_body]);

        let query = query_parser
            .parse_query(query_str)
            .context("Failed to parse search query")?;

        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;

        let mut results = Vec::with_capacity(top_docs.len());

        for (score, doc_address) in top_docs {
            let retrieved: TantivyDocument = searcher.doc(doc_address)?;

            let id = retrieved
                .get_first(self.f_id)
                .and_then(|v: &OwnedValue| v.as_str())
                .unwrap_or("")
                .to_string();

            let title = retrieved
                .get_first(self.f_title)
                .and_then(|v: &OwnedValue| v.as_str())
                .unwrap_or("")
                .to_string();

            let body = retrieved
                .get_first(self.f_body)
                .and_then(|v: &OwnedValue| v.as_str())
                .unwrap_or("")
                .to_string();

            let url = retrieved
                .get_first(self.f_url)
                .and_then(|v: &OwnedValue| v.as_str())
                .map(|s| if s.is_empty() { None } else { Some(s.to_string()) })
                .unwrap_or(None);

            let metadata = retrieved
                .get_first(self.f_metadata)
                .and_then(|v: &OwnedValue| v.as_str())
                .and_then(|s| {
                    if s.is_empty() {
                        None
                    } else {
                        serde_json::from_str(s).ok()
                    }
                });

            results.push(SearchResult {
                id,
                title,
                body,
                url,
                score,
                metadata,
            });
        }

        Ok(results)
    }

    /// Get a single document by ID.
    pub fn get_document(&self, id: &str) -> Result<Option<SearchResult>> {
        let results = self.search(&format!("\"{}\"", id), 1)?;
        // Filter to exact ID match
        Ok(results.into_iter().find(|r| r.id == id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_doc(id: &str, title: &str, body: &str) -> Document {
        Document {
            id: id.to_string(),
            title: title.to_string(),
            body: body.to_string(),
            url: None,
            metadata: None,
        }
    }

    #[test]
    fn test_add_and_search() {
        let dir = TempDir::new().unwrap();
        let index = SearchIndex::open_or_create(dir.path(), "test").unwrap();

        let docs = vec![
            make_doc("1", "Rust programming", "systems language with memory safety"),
            make_doc("2", "Python scripting", "dynamic language for data science"),
        ];
        index.add_documents(&docs).unwrap();
        index.commit().unwrap();

        let results = index.search("rust", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "1");
    }

    #[test]
    fn test_search_returns_empty_on_no_match() {
        let dir = TempDir::new().unwrap();
        let index = SearchIndex::open_or_create(dir.path(), "test").unwrap();
        index.add_documents(&[make_doc("1", "Hello world", "some text")]).unwrap();
        index.commit().unwrap();

        let results = index.search("xylophone", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_delete_document() {
        let dir = TempDir::new().unwrap();
        let index = SearchIndex::open_or_create(dir.path(), "test").unwrap();

        index.add_documents(&[make_doc("1", "Unique zephyr title", "unique zephyr body")]).unwrap();
        index.commit().unwrap();

        let before = index.search("zephyr", 10).unwrap();
        assert!(!before.is_empty());

        index.delete_document("1").unwrap();
        index.commit().unwrap();

        let after = index.search("zephyr", 10).unwrap();
        assert!(after.is_empty());
    }

    #[test]
    fn test_document_fields_preserved() {
        let dir = TempDir::new().unwrap();
        let index = SearchIndex::open_or_create(dir.path(), "test").unwrap();

        let mut doc = make_doc("42", "Field test", "checking stored fields");
        doc.url = Some("https://example.com".to_string());
        index.add_documents(&[doc]).unwrap();
        index.commit().unwrap();

        let results = index.search("checking", 10).unwrap();
        assert_eq!(results[0].url, Some("https://example.com".to_string()));
    }
}
