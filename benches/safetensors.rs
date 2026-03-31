//! SafeTensors benchmarks: tensor_store modes plus native baselines.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::{Path, PathBuf};
use tensor_store::formats::safetensors;

fn discover_fixtures() -> Vec<(String, PathBuf)> {
    let fixtures_dir = Path::new("fixtures");
    let mut fixtures = Vec::new();

    if let Ok(entries) = std::fs::read_dir(fixtures_dir) {
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if !file_type.is_dir() {
                continue;
            }

            let dir = entry.path();
            let has_safetensors = std::fs::read_dir(&dir)
                .ok()
                .into_iter()
                .flatten()
                .flatten()
                .any(|file| {
                    file.path()
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.ends_with(".safetensors"))
                });

            if has_safetensors {
                fixtures.push((entry.file_name().to_string_lossy().to_string(), dir));
            }
        }
    }

    fixtures.sort_by(|a, b| a.0.cmp(&b.0));
    fixtures
}

fn collect_shard_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if !file_type.is_file() {
                continue;
            }

            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.ends_with(".safetensors") {
                files.push(path);
            }
        }
    }
    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    files
}

fn bench_default(c: &mut Criterion) {
    let mut group = c.benchmark_group("safetensors_default");
    let rt = tokio::runtime::Runtime::new().unwrap();
    for (model_name, dir) in discover_fixtures() {
        let dir_str = dir.to_str().unwrap().to_string();
        group.bench_with_input(BenchmarkId::new("load", &model_name), &dir_str, |b, p| {
            b.to_async(&rt).iter(|| async {
                let data = safetensors::Model::load(black_box(p)).await.unwrap();
                black_box((data.len(), data.tensor_names().len()))
            });
        });
    }
    group.finish();
}

fn bench_sync(c: &mut Criterion) {
    let mut group = c.benchmark_group("safetensors_sync");
    for (model_name, dir) in discover_fixtures() {
        let dir_str = dir.to_str().unwrap().to_string();
        group.bench_with_input(BenchmarkId::new("load", &model_name), &dir_str, |b, p| {
            b.iter(|| {
                let data = safetensors::Model::load_sync(black_box(p)).unwrap();
                black_box((data.len(), data.tensor_names().len()))
            });
        });
    }
    group.finish();
}

fn bench_async(c: &mut Criterion) {
    let mut group = c.benchmark_group("safetensors_async");
    let rt = tokio::runtime::Runtime::new().unwrap();
    for (model_name, dir) in discover_fixtures() {
        let dir_str = dir.to_str().unwrap().to_string();
        group.bench_with_input(BenchmarkId::new("load", &model_name), &dir_str, |b, p| {
            b.to_async(&rt).iter(|| async {
                let data = safetensors::Model::load_async(black_box(p)).await.unwrap();
                black_box((data.len(), data.tensor_names().len()))
            });
        });
    }
    group.finish();
}

fn bench_mmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("safetensors_mmap");
    for (model_name, dir) in discover_fixtures() {
        let dir_str = dir.to_str().unwrap().to_string();
        group.bench_with_input(BenchmarkId::new("load", &model_name), &dir_str, |b, p| {
            b.iter(|| {
                let data = safetensors::MmapModel::open(black_box(p)).unwrap();
                black_box((data.len(), data.tensor_names().len()))
            });
        });
    }
    group.finish();
}

fn bench_native(c: &mut Criterion) {
    let mut group = c.benchmark_group("native_safetensors");
    for (model_name, dir) in discover_fixtures() {
        for file in collect_shard_files(&dir) {
            let file_name = file.file_name().unwrap().to_string_lossy().to_string();
            let path_str = file.to_str().unwrap().to_string();
            group.bench_with_input(
                BenchmarkId::new(&model_name, &file_name),
                &path_str,
                |b, p| {
                    b.iter(|| {
                        let bytes = std::fs::read(black_box(p)).unwrap();
                        let tensors = ::safetensors::SafeTensors::deserialize(&bytes).unwrap();
                        black_box((tensors.len(), bytes.len()))
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_default,
    bench_sync,
    bench_async,
    bench_mmap,
    bench_native
);
criterion_main!(benches);
