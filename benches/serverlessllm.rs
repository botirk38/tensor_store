//! ServerlessLLM benchmarks per backend: sync, mmap, async.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::PathBuf;
use tensor_store::TensorMetadata;
use tensor_store::formats::serverlessllm;

fn touch_pages(data: &[u8]) -> u8 {
    const PAGE: usize = 4096;
    if data.is_empty() {
        return 0;
    }
    let mut idx = 0;
    let mut checksum = 0u8;
    while idx < data.len() {
        checksum ^= data[idx];
        idx += PAGE;
    }
    checksum ^ data[data.len() - 1]
}

fn discover_fixtures() -> Vec<(String, PathBuf)> {
    let fixtures_dir = std::path::Path::new("fixtures");
    let mut fixtures = Vec::new();
    if let Ok(entries) = std::fs::read_dir(fixtures_dir) {
        for entry in entries.flatten() {
            if let Ok(file_type) = entry.file_type()
                && file_type.is_dir()
            {
                let model_dir = entry.path().join("model_serverlessllm");
                if model_dir.exists() && model_dir.is_dir() {
                    let model_name = entry.file_name().to_string_lossy().to_string();
                    fixtures.push((model_name, model_dir));
                }
            }
        }
    }
    fixtures.sort_by(|a, b| a.0.cmp(&b.0));
    fixtures
}

fn bench_sync(c: &mut Criterion) {
    let mut group = c.benchmark_group("serverlessllm_sync");
    for (model_name, dir) in discover_fixtures() {
        let dir_str = dir.to_str().unwrap().to_string();
        group.bench_with_input(BenchmarkId::new("load", &model_name), &dir_str, |b, p| {
            b.iter(|| {
                let model = serverlessllm::load_sync(black_box(p)).unwrap();
                let n = model.len();
                let bytes: usize = model.iter().map(|(_, t)| t.data().len()).sum();
                black_box((n, bytes))
            });
        });
    }
    group.finish();
}

fn bench_mmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("serverlessllm_mmap");
    for (model_name, dir) in discover_fixtures() {
        let dir_str = dir.to_str().unwrap().to_string();
        group.bench_with_input(BenchmarkId::new("load", &model_name), &dir_str, |b, p| {
            b.iter(|| {
                let model = serverlessllm::load_mmap(black_box(p)).unwrap();
                let names = model.tensor_names();
                let mut checksum = 0u8;
                let mut bytes = 0;
                for name in &names {
                    let tensor = model.tensor(name).unwrap();
                    let data = tensor.data();
                    bytes += data.len();
                    checksum ^= touch_pages(data);
                }
                black_box((model, bytes, checksum))
            });
        });
    }
    group.finish();
}

#[cfg(target_os = "linux")]
fn bench_async(c: &mut Criterion) {
    let mut group = c.benchmark_group("serverlessllm_async");
    for (model_name, dir) in discover_fixtures() {
        let dir_str = dir.to_str().unwrap().to_string();
        group.bench_with_input(BenchmarkId::new("load", &model_name), &dir_str, |b, p| {
            b.iter(|| {
                tokio_uring::start(async {
                    let model = serverlessllm::load(black_box(p)).await.unwrap();
                    let n = model.len();
                    let bytes: usize = model.iter().map(|(_, t)| t.data().len()).sum();
                    black_box((n, bytes))
                })
            });
        });
    }
    group.finish();
}

#[cfg(not(target_os = "linux"))]
fn bench_async(c: &mut Criterion) {
    let mut group = c.benchmark_group("serverlessllm_async");
    let rt = tokio::runtime::Runtime::new().unwrap();
    for (model_name, dir) in discover_fixtures() {
        let dir_str = dir.to_str().unwrap().to_string();
        group.bench_with_input(BenchmarkId::new("load", &model_name), &dir_str, |b, p| {
            b.to_async(&rt).iter(|| async {
                let model = serverlessllm::load(black_box(p)).await.unwrap();
                let n = model.len();
                let bytes: usize = model.iter().map(|(_, t)| t.data().len()).sum();
                black_box((n, bytes))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sync, bench_mmap, bench_async);
criterion_main!(benches);
