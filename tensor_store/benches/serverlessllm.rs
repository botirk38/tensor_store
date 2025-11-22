use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tensor_store::readers::serverlessllm;
use tensor_store::TensorMetadata;

#[cfg(target_os = "linux")]
fn bench_io_uring(c: &mut Criterion) {
    let serverlessllm_dir = "test_model_serverlessllm";

    // ServerlessLLM io_uring loading (load all tensors sequentially)
    c.bench_function("io_uring_serverlessllm_all_tensors", |b| {
        b.iter(|| {
            tokio_uring::start(async {
                let index_path = format!("{}/tensor_index.json", black_box(serverlessllm_dir));
                let data_path = format!("{}/tensor.data", black_box(serverlessllm_dir));

                let index = serverlessllm::parse_index(&index_path).await.unwrap();
                let tensor_names = index.tensor_names();
                let tensor_count = tensor_names.len();

                let mut total_bytes = 0;
                for name in tensor_names {
                    let tensor_data = index.load_tensor(&data_path, name).await.unwrap();
                    total_bytes += tensor_data.len();
                }

                black_box((total_bytes, tensor_count))
            })
        });
    });

    // ServerlessLLM io_uring loading (load all partitions in parallel)
    c.bench_function("io_uring_serverlessllm_parallel_partitions", |b| {
        b.iter(|| {
            tokio_uring::start(async {
                use std::collections::HashMap;

                let index_path = format!("{}/tensor_index.json", black_box(serverlessllm_dir));
                let data_path = format!("{}/tensor.data", black_box(serverlessllm_dir));

                let index = serverlessllm::parse_index(&index_path).await.unwrap();
                let tensor_names = index.tensor_names();
                let tensor_count = tensor_names.len();

                // Group tensors by partition
                let mut partition_groups: HashMap<usize, Vec<&str>> = HashMap::new();
                for name in &tensor_names {
                    if let Some(entry) = index.get(name) {
                        partition_groups
                            .entry(entry.partition_id)
                            .or_default()
                            .push(name);
                    }
                }

                // Load all partitions in parallel
                let mut handles = Vec::new();
                for (partition_id, _) in partition_groups {
                    let data_path_clone = data_path.clone();

                    handles.push(tokio_uring::spawn(async move {
                        // Open partition file directly instead of loading through index
                        let partition_path = format!("{}_{}", data_path_clone, partition_id);
                        let file = tokio_uring::fs::File::open(&partition_path).await.unwrap();
                        let metadata = file.statx().await.unwrap();
                        let file_size = metadata.stx_size;

                        // Read entire partition file
                        let buf = vec![0u8; file_size as usize];
                        let (res, buf) = file.read_at(buf, 0).await;
                        res.unwrap();

                        buf.len()
                    }));
                }

                let mut total_bytes = 0;
                for handle in handles {
                    total_bytes += handle.await.unwrap();
                }

                black_box((total_bytes, tensor_count))
            })
        });
    });
}

#[cfg(not(target_os = "linux"))]
fn bench_tokio(c: &mut Criterion) {
    let serverlessllm_dir = "test_model_serverlessllm";
    let rt = tokio::runtime::Runtime::new().unwrap();

    // ServerlessLLM tokio loading (load all tensors sequentially)
    c.bench_function("tokio_serverlessllm_all_tensors", |b| {
        b.to_async(&rt).iter(|| async {
            let index_path = format!("{}/tensor_index.json", black_box(serverlessllm_dir));
            let data_path = format!("{}/tensor.data", black_box(serverlessllm_dir));

            let index = serverlessllm::parse_index(&index_path).await.unwrap();
            let tensor_names = index.tensor_names();
            let tensor_count = tensor_names.len();

            let mut total_bytes = 0;
            for name in tensor_names {
                let tensor_data = index.load_tensor(&data_path, name).await.unwrap();
                total_bytes += tensor_data.len();
            }

            black_box((total_bytes, tensor_count))
        });
    });

    // ServerlessLLM tokio loading (load all partitions in parallel)
    c.bench_function("tokio_serverlessllm_parallel_partitions", |b| {
        b.to_async(&rt).iter(|| async {
            use std::collections::HashMap;
            use tokio::fs;
            use tokio::io::AsyncReadExt;

            let index_path = format!("{}/tensor_index.json", black_box(serverlessllm_dir));
            let data_path = format!("{}/tensor.data", black_box(serverlessllm_dir));

            let index = serverlessllm::parse_index(&index_path).await.unwrap();
            let tensor_names = index.tensor_names();
            let tensor_count = tensor_names.len();

            // Group tensors by partition
            let mut partition_groups: HashMap<usize, Vec<&str>> = HashMap::new();
            for name in &tensor_names {
                if let Some(entry) = index.get(name) {
                    partition_groups
                        .entry(entry.partition_id)
                        .or_insert_with(Vec::new)
                        .push(name);
                }
            }

            // Load all partitions in parallel
            let mut handles = Vec::new();
            for (partition_id, _) in partition_groups {
                let data_path_clone = data_path.clone();

                handles.push(tokio::spawn(async move {
                    // Open partition file directly instead of loading through index
                    let partition_path = format!("{}_{}", data_path_clone, partition_id);
                    let mut file = fs::File::open(&partition_path).await.unwrap();
                    let metadata = file.metadata().await.unwrap();
                    let file_size = metadata.len();

                    // Read entire partition file
                    let mut buf = vec![0u8; file_size as usize];
                    file.read_exact(&mut buf).await.unwrap();

                    buf.len()
                }));
            }

            let mut total_bytes = 0;
            for handle in handles {
                total_bytes += handle.await.unwrap();
            }

            black_box((total_bytes, tensor_count))
        });
    });
}

fn bench_sync_serverlessllm(c: &mut Criterion) {
    let serverlessllm_dir = "test_model_serverlessllm";

    // ServerlessLLM sync loading (load all tensors)
    c.bench_function("sync_serverlessllm_all_tensors", |b| {
        b.iter(|| {
            let index_path = format!("{}/tensor_index.json", black_box(serverlessllm_dir));
            let data_path = format!("{}/tensor.data", black_box(serverlessllm_dir));

            let index = serverlessllm::parse_index_sync(&index_path).unwrap();
            let tensor_names = index.tensor_names();
            let tensor_count = tensor_names.len();

            let mut total_bytes = 0;
            for name in tensor_names {
                let tensor_data = index.load_tensor_sync(&data_path, name).unwrap();
                total_bytes += tensor_data.len();
            }

            black_box((total_bytes, tensor_count))
        });
    });
}

#[cfg(target_os = "linux")]
criterion_group!(benches, bench_io_uring, bench_sync_serverlessllm);

#[cfg(not(target_os = "linux"))]
criterion_group!(benches, bench_tokio, bench_sync_serverlessllm);

criterion_main!(benches);
