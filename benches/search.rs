use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kakugosearch::index::search_index::{Document, SearchIndex};
use std::time::Duration;
use tempfile::TempDir;

// ── Synthetic data ────────────────────────────────────────────────────────────

const TOPICS: &[&str] = &[
    "machine learning",
    "distributed systems",
    "web development",
    "database internals",
    "compiler design",
    "operating systems",
    "cryptography",
    "computer vision",
    "natural language processing",
    "systems programming",
];

const ADJECTIVES: &[&str] = &[
    "performance", "scalability", "reliability", "concurrency",
    "memory safety", "throughput", "latency", "algorithms",
    "data structures", "optimization",
];

fn make_doc(i: usize) -> Document {
    let topic = TOPICS[i % TOPICS.len()];
    let adj = ADJECTIVES[i % ADJECTIVES.len()];
    let adj2 = ADJECTIVES[(i + 3) % ADJECTIVES.len()];
    Document {
        id: i.to_string(),
        title: format!("Document {i}: {topic}"),
        body: format!(
            "This document covers {topic} in depth, focusing on {adj} and {adj2}. \
             Modern {topic} systems must balance {adj} with correctness guarantees. \
             Practitioners in {topic} often encounter trade-offs between {adj} and {adj2}.",
        ),
        url: Some(format!("https://example.com/docs/{i}")),
        metadata: None,
    }
}

fn make_docs(n: usize) -> Vec<Document> {
    (0..n).map(make_doc).collect()
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

/// How fast can we ingest and commit batches of documents?
/// Reports throughput in docs/sec for batch sizes of 100, 1 000, and 10 000.
fn bench_index_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("indexing_throughput");
    // Each iteration takes ~230 ms. 10 samples gives stable stats without
    // blowing the default time budget.
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for batch_size in [100_usize, 1_000, 10_000] {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("docs", batch_size),
            &batch_size,
            |b, &size| {
                let docs = make_docs(size);
                b.iter_batched(
                    || {
                        let dir = TempDir::new().unwrap();
                        let index =
                            SearchIndex::open_or_create(dir.path(), "bench").unwrap();
                        (dir, index, docs.clone())
                    },
                    |(_, index, docs)| {
                        index.add_documents(&docs).unwrap();
                        index.commit().unwrap();
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

/// Single-threaded search latency on a warm 10 000-document index.
/// Exercises several query types: single term, phrase, multi-word.
fn bench_search_warm(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let index = SearchIndex::open_or_create(dir.path(), "bench").unwrap();
    index.add_documents(&make_docs(10_000)).unwrap();
    index.commit().unwrap();
    // Tantivy's OnCommitWithDelay reader schedules a background reload after
    // commit. Sleep briefly so it fires before timing starts, preventing it
    // from appearing as an outlier mid-measurement.
    std::thread::sleep(Duration::from_millis(200));

    let queries = [
        "machine learning",
        "distributed systems performance",
        "memory safety concurrency",
        "database query optimization",
        "natural language processing algorithms",
    ];

    let mut group = c.benchmark_group("search_warm_10k");
    // Accept up to 5% noise — search latency has inherent OS scheduler jitter.
    group.noise_threshold(0.05);

    for query in &queries {
        group.bench_with_input(BenchmarkId::from_parameter(query), query, |b, q| {
            b.iter(|| index.search(q, 20).unwrap());
        });
    }
    group.finish();
}

/// How does search latency grow as the index grows?
/// Measures the same query on indexes of 1 K, 5 K, and 10 K documents.
fn bench_search_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_scaling");
    group.sample_size(20); // reduce samples — setup dominates
    group.noise_threshold(0.05);

    for doc_count in [1_000_usize, 5_000, 10_000] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("docs", doc_count),
            &doc_count,
            |b, &count| {
                let dir = TempDir::new().unwrap();
                let index =
                    SearchIndex::open_or_create(dir.path(), "bench").unwrap();
                index.add_documents(&make_docs(count)).unwrap();
                index.commit().unwrap();
                std::thread::sleep(Duration::from_millis(200));

                b.iter(|| index.search("machine learning algorithms", 20).unwrap());
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_index_throughput,
    bench_search_warm,
    bench_search_scaling,
);
criterion_main!(benches);
