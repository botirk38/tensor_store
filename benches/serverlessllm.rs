//! ServerlessLLM benchmarks: default, sync, async, mmap.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::{Path, PathBuf};
use tensor_store::formats::serverlessllm;

fn discover_fixtures() -> Vec<(String, PathBuf)> {
    let fixtures_dir = Path::new("fixtures");
    let mut fixtures = Vec::new();
    if let Ok(entries) = std::fs::read_dir(fixtures_dir) {
        for entry in entries.flatten() {
            if let Ok(file_type) = entry.file_type()
                && file_type.is_dir()
            {
                let model_dir = entry.path().join("model_serverlessllm");
                if model_dir.exists() && model_dir.is_dir() {
                    fixtures.push((entry.file_name().to_string_lossy().to_string(), model_dir));
                }
            }
        }
    }
    fixtures.sort_by(|a, b| a.0.cmp(&b.0));
    fixtures
}

fn bench_default(c: &mut Criterion) {
    let mut group = c.benchmark_group("serverlessllm_default");
    let rt = tokio::runtime::Runtime::new().unwrap();
    for (model_name, dir) in discover_fixtures() {
        let dir_str = dir.to_str().unwrap().to_string();
        group.bench_with_input(BenchmarkId::new("load", &model_name), &dir_str, |b, p| {
            b.to_async(&rt).iter(|| async {
                let model = serverlessllm::Model::load(black_box(p)).await.unwrap();
                let bytes: usize = (&model).into_iter().map(|(_, t)| t.data().len()).sum();
                black_box((model.len(), bytes))
            });
        });
    }
    group.finish();
}

fn bench_sync(c: &mut Criterion) {
    let mut group = c.benchmark_group("serverlessllm_sync");
    for (model_name, dir) in discover_fixtures() {
        let dir_str = dir.to_str().unwrap().to_string();
        group.bench_with_input(BenchmarkId::new("load", &model_name), &dir_str, |b, p| {
            b.iter(|| {
                let model = serverlessllm::Model::load_sync(black_box(p)).unwrap();
                let bytes: usize = (&model).into_iter().map(|(_, t)| t.data().len()).sum();
                black_box((model.len(), bytes))
            });
        });
    }
    group.finish();
}

fn bench_async(c: &mut Criterion) {
    let mut group = c.benchmark_group("serverlessllm_async");
    let rt = tokio::runtime::Runtime::new().unwrap();
    for (model_name, dir) in discover_fixtures() {
        let dir_str = dir.to_str().unwrap().to_string();
        group.bench_with_input(BenchmarkId::new("load", &model_name), &dir_str, |b, p| {
            b.to_async(&rt).iter(|| async {
                let model = serverlessllm::Model::load_async(black_box(p))
                    .await
                    .unwrap();
                let bytes: usize = (&model).into_iter().map(|(_, t)| t.data().len()).sum();
                black_box((model.len(), bytes))
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
                let model = serverlessllm::MmapModel::open(black_box(p)).unwrap();
                let bytes: usize = model
                    .tensor_names()
                    .iter()
                    .map(|name| model.tensor(name).unwrap().data().len())
                    .sum();
                black_box((model.len(), bytes))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_default, bench_sync, bench_async, bench_mmap);
criterion_main!(benches);
