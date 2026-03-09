//! SafeTensors benchmarks per backend: sync, mmap, async.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::PathBuf;
use tensor_store::formats::safetensors;

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
                let model_path = entry.path().join("model.safetensors");
                if model_path.exists() {
                    let model_name = entry.file_name().to_string_lossy().to_string();
                    fixtures.push((model_name, model_path));
                }
            }
        }
    }
    fixtures.sort_by(|a, b| a.0.cmp(&b.0));
    fixtures
}

fn bench_sync(c: &mut Criterion) {
    let mut group = c.benchmark_group("safetensors_sync");
    for (model_name, path) in discover_fixtures() {
        let path_str = path.to_str().unwrap().to_string();
        group.bench_with_input(BenchmarkId::new("load", &model_name), &path_str, |b, p| {
            b.iter(|| {
                let data = safetensors::Model::load_sync(black_box(p)).unwrap();
                black_box((data.names().len(), data.into_bytes()))
            });
        });
    }
    group.finish();
}

fn bench_mmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("safetensors_mmap");
    for (model_name, path) in discover_fixtures() {
        let path_str = path.to_str().unwrap().to_string();
        group.bench_with_input(BenchmarkId::new("load", &model_name), &path_str, |b, p| {
            b.iter(|| {
                let data = safetensors::MmapModel::load(black_box(p)).unwrap();
                let tensors = data.tensors();
                let names = tensors.names();
                let mut checksum = 0u8;
                for name in names {
                    checksum ^= touch_pages(tensors.tensor(name).unwrap().data());
                }
                black_box((data, checksum))
            });
        });
    }
    group.finish();
}

#[cfg(target_os = "linux")]
fn bench_async(c: &mut Criterion) {
    let mut group = c.benchmark_group("safetensors_async");
    for (model_name, path) in discover_fixtures() {
        let path_str = path.to_str().unwrap().to_string();
        group.bench_with_input(BenchmarkId::new("load", &model_name), &path_str, |b, p| {
            b.iter(|| {
                tokio_uring::start(async {
                    let data = safetensors::Model::load(black_box(p)).await.unwrap();
                    black_box((data.names().len(), data))
                })
            });
        });
    }
    group.finish();
}

#[cfg(target_os = "linux")]
fn bench_async_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("safetensors_async_parallel");
    let num_cores = num_cpus::get();
    for (model_name, path) in discover_fixtures() {
        let path_str = path.to_str().unwrap().to_string();
        group.bench_with_input(
            BenchmarkId::new("load", format!("{}_cores_{}", num_cores, model_name)),
            &path_str,
            |b, p| {
                b.iter(|| {
                    tokio_uring::start(async {
                        let data = safetensors::Model::load_parallel(black_box(p), num_cores)
                            .await
                            .unwrap();
                        black_box((data.tensors().names().len(), data))
                    })
                });
            },
        );
    }
    group.finish();
}

#[cfg(not(target_os = "linux"))]
fn bench_async(c: &mut Criterion) {
    let mut group = c.benchmark_group("safetensors_async");
    let rt = tokio::runtime::Runtime::new().unwrap();
    for (model_name, path) in discover_fixtures() {
        let path_str = path.to_str().unwrap().to_string();
        group.bench_with_input(BenchmarkId::new("load", &model_name), &path_str, |b, p| {
            b.to_async(&rt).iter(|| async {
                let data = safetensors::Model::load(black_box(p)).await.unwrap();
                black_box((data.names().len(), data))
            });
        });
    }
    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    benches,
    bench_sync,
    bench_mmap,
    bench_async,
    bench_async_parallel
);

#[cfg(not(target_os = "linux"))]
criterion_group!(benches, bench_sync, bench_mmap, bench_async);

criterion_main!(benches);
