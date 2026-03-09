/// KakugoSearch concurrent load test.
///
/// Spins up N tokio threads, each firing searches in a tight loop for a fixed
/// duration, then reports QPS and latency percentiles (p50 / p95 / p99).
///
/// Run:
///   cargo run --example load_test --release
use hdrhistogram::Histogram;
use kakugosearch::index::search_index::{Document, SearchIndex};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::TempDir;

const INDEX_SIZE: usize = 10_000;
const TEST_DURATION: Duration = Duration::from_secs(5);

const QUERIES: &[&str] = &[
    "machine learning",
    "distributed systems performance",
    "memory safety concurrency",
    "database query optimization",
    "natural language processing algorithms",
];

fn make_doc(i: usize) -> Document {
    let topics = [
        "machine learning", "distributed systems", "compiler design",
        "operating systems", "cryptography", "computer vision",
        "natural language processing", "systems programming",
    ];
    let attrs = [
        "memory safety", "concurrency", "optimization", "algorithms",
        "data structures", "cryptography", "computer vision",
    ];
    let topic = topics[i % topics.len()];
    let attr = attrs[i % attrs.len()];
    Document {
        id: i.to_string(),
        title: format!("Document {i}: {topic}"),
        body: format!(
            "This document covers {topic} and {attr}. \
             Performance, scalability, and reliability are key concerns in {topic}. \
             Modern {attr} techniques are widely applied in {topic} systems.",
        ),
        url: Some(format!("https://example.com/doc/{i}")),
        metadata: None,
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Run `concurrency` parallel reader tasks for `TEST_DURATION`, return (qps, p50_ms, p95_ms, p99_ms).
async fn run_readers(index: Arc<SearchIndex>, concurrency: usize) -> (f64, f64, f64, f64) {
    let start = Instant::now();
    let deadline = start + TEST_DURATION;

    let handles: Vec<_> = (0..concurrency)
        .map(|i| {
            let index = index.clone();
            tokio::task::spawn_blocking(move || {
                let mut hist = Histogram::<u64>::new(3).unwrap();
                while Instant::now() < deadline {
                    let query = QUERIES[i % QUERIES.len()];
                    let t = Instant::now();
                    index.search(query, 20).unwrap();
                    hist.record(t.elapsed().as_micros() as u64).unwrap();
                }
                hist
            })
        })
        .collect();

    let mut combined = Histogram::<u64>::new(3).unwrap();
    for h in handles {
        combined.add(h.await.unwrap()).unwrap();
    }
    let elapsed = start.elapsed();

    let qps = combined.len() as f64 / elapsed.as_secs_f64();
    let p50 = combined.value_at_quantile(0.50) as f64 / 1_000.0;
    let p95 = combined.value_at_quantile(0.95) as f64 / 1_000.0;
    let p99 = combined.value_at_quantile(0.99) as f64 / 1_000.0;
    (qps, p50, p95, p99)
}

/// 8 reader tasks + 2 writer tasks running concurrently.
async fn run_mixed(index: Arc<SearchIndex>) -> (f64, f64, f64, f64) {
    let start = Instant::now();
    let deadline = start + TEST_DURATION;

    // writers — small batches, commit after each
    let writer_handles: Vec<_> = (0..2_usize)
        .map(|j| {
            let index = index.clone();
            let base = INDEX_SIZE + j * 500_000;
            tokio::task::spawn_blocking(move || {
                let mut id = base;
                while Instant::now() < deadline {
                    let batch: Vec<Document> = (id..id + 10).map(make_doc).collect();
                    index.add_documents(&batch).unwrap();
                    index.commit().unwrap();
                    id += 10;
                }
            })
        })
        .collect();

    // readers — measure latency
    let reader_handles: Vec<_> = (0..8_usize)
        .map(|i| {
            let index = index.clone();
            tokio::task::spawn_blocking(move || {
                let mut hist = Histogram::<u64>::new(3).unwrap();
                while Instant::now() < deadline {
                    let query = QUERIES[i % QUERIES.len()];
                    let t = Instant::now();
                    index.search(query, 20).unwrap();
                    hist.record(t.elapsed().as_micros() as u64).unwrap();
                }
                hist
            })
        })
        .collect();

    let mut combined = Histogram::<u64>::new(3).unwrap();
    for h in reader_handles {
        combined.add(h.await.unwrap()).unwrap();
    }
    for h in writer_handles {
        h.await.unwrap();
    }
    let elapsed = start.elapsed();

    let qps = combined.len() as f64 / elapsed.as_secs_f64();
    let p50 = combined.value_at_quantile(0.50) as f64 / 1_000.0;
    let p95 = combined.value_at_quantile(0.95) as f64 / 1_000.0;
    let p99 = combined.value_at_quantile(0.99) as f64 / 1_000.0;
    (qps, p50, p95, p99)
}

// ── main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    println!("KakugoSearch — Concurrent Load Test");
    println!("====================================");
    println!("Building index with {INDEX_SIZE} documents...");

    let dir = TempDir::new().unwrap();
    let index = Arc::new(SearchIndex::open_or_create(dir.path(), "bench").unwrap());
    let docs: Vec<Document> = (0..INDEX_SIZE).map(make_doc).collect();
    index.add_documents(&docs).unwrap();
    index.commit().unwrap();

    println!("Index ready. Each test runs for {}s.\n", TEST_DURATION.as_secs());

    // ── pure read scaling ─────────────────────────────────────────────────────
    println!(
        "{:<14} {:>10} {:>10} {:>10} {:>10}",
        "Concurrency", "QPS", "p50 ms", "p95 ms", "p99 ms"
    );
    println!("{}", "─".repeat(58));

    for concurrency in [1_usize, 4, 8, 16, 32] {
        let (qps, p50, p95, p99) = run_readers(index.clone(), concurrency).await;
        println!(
            "{:<14} {:>10.0} {:>10.2} {:>10.2} {:>10.2}",
            concurrency, qps, p50, p95, p99
        );
    }

    // ── mixed read + write ────────────────────────────────────────────────────
    println!();
    println!("Mixed (8 readers + 2 writers):");
    let (qps, p50, p95, p99) = run_mixed(index.clone()).await;
    println!(
        "  QPS={qps:.0}  p50={p50:.2}ms  p95={p95:.2}ms  p99={p99:.2}ms"
    );
}
