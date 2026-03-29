use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};

use ::safetensors::SafeTensors;
use tensor_store::formats::safetensors;

use crate::config::{ProfileConfig, ProfileError, ProfileResult};
use crate::stats::summarize;

#[cfg(unix)]
fn drop_page_cache(path: &Path) {
    use std::os::unix::io::AsRawFd;
    if let Ok(file) = std::fs::File::open(path) {
        let fd = file.as_raw_fd();
        unsafe {
            let _ = libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
        }
    }
}

#[cfg(not(unix))]
fn drop_page_cache(_path: &Path) {
    // No-op on non-unix
}

pub fn run(case: &str, config: &ProfileConfig) -> ProfileResult {
    match case {
        "io-uring-load" => io_uring_load(config),
        "io-uring-prewarmed" => io_uring_prewarmed(config),
        "tokio-load" => tokio_load(config),
        "tokio-prewarmed" => tokio_prewarmed(config),
        "sync" => sync_load(config),
        "mmap" => mmap_load(config),
        "original" => original_load(config),
        other => Err(ProfileError::new(format!("Unknown safetensors case '{}'", other)).into()),
    }
}

fn collect_safetensors_files(dir: &Path) -> Vec<PathBuf> {
    let mut shard_files = Vec::new();
    let mut single_file = None;

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

            if name == "model.safetensors" {
                single_file = Some(path);
            } else if name.ends_with(".safetensors") {
                shard_files.push(path);
            }
        }
    }

    if !shard_files.is_empty() {
        shard_files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
        return shard_files;
    }

    single_file.into_iter().collect()
}

fn discover_fixtures() -> Vec<(String, Vec<PathBuf>)> {
    let fixtures_dir = std::path::Path::new("fixtures");
    let mut fixtures = Vec::new();

    if let Ok(entries) = std::fs::read_dir(fixtures_dir) {
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if !file_type.is_dir() {
                continue;
            }

            let model_files = collect_safetensors_files(&entry.path());
            if !model_files.is_empty() {
                let model_name = entry.file_name().to_string_lossy().to_string();
                fixtures.push((model_name, model_files));
            }
        }
    }

    fixtures.sort_by(|a, b| a.0.cmp(&b.0));
    fixtures
}

fn fixtures(config: &ProfileConfig) -> Result<Vec<(String, Vec<PathBuf>)>, ProfileError> {
    let fixtures = discover_fixtures();
    if fixtures.is_empty() {
        return Err(ProfileError::new(
            "No safetensors fixtures found under 'fixtures/'.",
        ));
    }

    if let Some(name) = &config.fixture {
        let filtered: Vec<_> = fixtures
            .into_iter()
            .filter(|(fixture_name, _)| fixture_name == name)
            .collect();
        if filtered.is_empty() {
            return Err(ProfileError::new(format!(
                "Fixture '{}' not found. Available: {}",
                name,
                discover_fixtures()
                    .into_iter()
                    .map(|(n, _)| n)
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        Ok(filtered)
    } else {
        Ok(fixtures)
    }
}

fn total_file_bytes(files: &[PathBuf]) -> std::io::Result<u64> {
    let mut total = 0u64;
    for path in files {
        total += fs::metadata(path)?.len();
    }
    Ok(total)
}

fn drop_page_cache_for_files(files: &[PathBuf]) {
    for path in files {
        drop_page_cache(path);
    }
}

fn path_to_string(path: &Path) -> std::io::Result<String> {
    path.to_str().map(|s| s.to_owned()).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Fixture path contains invalid UTF-8",
        )
    })
}

#[cfg(target_os = "linux")]
fn io_uring_load(config: &ProfileConfig) -> ProfileResult {
    use std::time::Instant;
    let fixtures = fixtures(config)?;

    for (fixture, files) in fixtures {
        let iterations = config.normalized_iterations();
        let cache_label = if config.cold_cache { "cold" } else { "warm" };
        let mut durations = Vec::with_capacity(iterations);
        let total_bytes = total_file_bytes(&files)?;

        println!(
            "Running io_uring safetensors load for '{}' ({iterations}x {cache_label})",
            fixture
        );
        tokio_uring::start(async {
            for i in 0..iterations {
                if config.cold_cache && i == 0 {
                    drop_page_cache_for_files(&files);
                }

                let start = Instant::now();
                let mut tensor_count = 0usize;
                for path in &files {
                    let path_str = path_to_string(path)?;
                    let data = safetensors::Model::load(&path_str).await?;
                    tensor_count += data.names().len();
                }
                let elapsed = start.elapsed();
                durations.push(elapsed);
                println!(
                    "  iteration {}: {} tensors, {} bytes, {:.2}ms",
                    i + 1,
                    tensor_count,
                    total_bytes,
                    elapsed.as_secs_f64() * 1000.0
                );
                black_box((tensor_count, total_bytes));
            }

            if let Some(summary) = summarize(&durations) {
                println!(
                    "  summary: mean {:.2}ms | min {:.2}ms | max {:.2}ms",
                    summary.mean_ms, summary.min_ms, summary.max_ms
                );
            }
            Ok::<_, tensor_store::ReaderError>(())
        })?;
    }

    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn io_uring_load(_config: &ProfileConfig) -> ProfileResult {
    Err(ProfileError::new("io_uring safetensors cases are only available on Linux").into())
}

#[cfg(target_os = "linux")]
fn io_uring_prewarmed(config: &ProfileConfig) -> ProfileResult {
    let fixtures = fixtures(config)?;

    for (fixture, files) in fixtures {
        let iterations = config.normalized_iterations();
        let cache_label = if config.cold_cache { "cold" } else { "warm" };
        let mut durations = Vec::with_capacity(iterations);
        let total_bytes = total_file_bytes(&files)?;

        println!(
            "Running io_uring safetensors prewarmed load for '{}' ({iterations}x {cache_label})",
            fixture
        );
        tokio_uring::start(async {
            for _ in 0..2 {
                for path in &files {
                    let path_str = path_to_string(path)?;
                    let _ = safetensors::Model::load(&path_str).await?;
                }
            }

            for i in 0..iterations {
                if config.cold_cache && i == 0 {
                    drop_page_cache_for_files(&files);
                }

                let start = std::time::Instant::now();
                let mut tensor_count = 0usize;
                for path in &files {
                    let path_str = path_to_string(path)?;
                    let data = safetensors::Model::load(&path_str).await?;
                    tensor_count += data.names().len();
                }
                let elapsed = start.elapsed();
                durations.push(elapsed);
                black_box((total_bytes, tensor_count));
            }

            if let Some(summary) = summarize(&durations) {
                println!(
                    "  summary: mean {:.2}ms | min {:.2}ms | max {:.2}ms",
                    summary.mean_ms, summary.min_ms, summary.max_ms
                );
            }
            Ok::<_, tensor_store::ReaderError>(())
        })?;
    }

    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn io_uring_prewarmed(_config: &ProfileConfig) -> ProfileResult {
    Err(ProfileError::new("io_uring safetensors cases are only available on Linux").into())
}

#[cfg(not(target_os = "linux"))]
fn tokio_load(config: &ProfileConfig) -> ProfileResult {
    use std::time::Instant;
    let fixtures = fixtures(config)?;
    let rt = tokio::runtime::Runtime::new()?;

    for (fixture, files) in fixtures {
        let iterations = config.normalized_iterations();
        let cache_label = if config.cold_cache { "cold" } else { "warm" };
        let mut durations = Vec::with_capacity(iterations);
        let total_bytes = total_file_bytes(&files)?;

        println!(
            "Running tokio safetensors load for '{}' ({iterations}x {cache_label})",
            fixture
        );
        rt.block_on(async {
            for i in 0..iterations {
                if config.cold_cache && i == 0 {
                    drop_page_cache_for_files(&files);
                }
                let start = Instant::now();
                let mut tensor_count = 0usize;
                for path in &files {
                    let path_str = path_to_string(path)?;
                    let data = safetensors::Model::load(&path_str).await?;
                    tensor_count += data.names().len();
                }
                let elapsed = start.elapsed();
                durations.push(elapsed);
                println!(
                    "  iteration {}: {} tensors, {} bytes, {:.2}ms",
                    i + 1,
                    tensor_count,
                    bytes,
                    elapsed.as_secs_f64() * 1000.0
                );
                black_box((total_bytes, tensor_count));
            }

            if let Some(summary) = summarize(&durations) {
                println!(
                    "  summary: mean {:.2}ms | min {:.2}ms | max {:.2}ms",
                    summary.mean_ms, summary.min_ms, summary.max_ms
                );
            }
            Ok::<_, tensor_store::ReaderError>(())
        })?;
    }

    Ok(())
}

#[cfg(target_os = "linux")]
fn tokio_load(_config: &ProfileConfig) -> ProfileResult {
    Err(ProfileError::new(
        "tokio safetensors cases are only compiled on non-Linux targets in this harness",
    )
    .into())
}

#[cfg(not(target_os = "linux"))]
fn tokio_prewarmed(config: &ProfileConfig) -> ProfileResult {
    let fixtures = fixtures(config)?;
    let rt = tokio::runtime::Runtime::new()?;

    for (fixture, files) in fixtures {
        let iterations = config.normalized_iterations();
        let cache_label = if config.cold_cache { "cold" } else { "warm" };
        let mut durations = Vec::with_capacity(iterations);
        let total_bytes = total_file_bytes(&files)?;

        println!(
            "Running tokio safetensors prewarmed load for '{}' ({iterations}x {cache_label})",
            fixture
        );
        rt.block_on(async {
            for _ in 0..2 {
                for path in &files {
                    let path_str = path_to_string(path)?;
                    let _ = safetensors::Model::load(&path_str).await?;
                }
            }

            for i in 0..iterations {
                if config.cold_cache && i == 0 {
                    drop_page_cache_for_files(&files);
                }
                let start = std::time::Instant::now();
                let mut tensor_count = 0usize;
                for path in &files {
                    let path_str = path_to_string(path)?;
                    let data = safetensors::Model::load(&path_str).await?;
                    tensor_count += data.names().len();
                }
                let elapsed = start.elapsed();
                durations.push(elapsed);
                black_box((total_bytes, tensor_count));
            }

            if let Some(summary) = summarize(&durations) {
                println!(
                    "  summary: mean {:.2}ms | min {:.2}ms | max {:.2}ms",
                    summary.mean_ms, summary.min_ms, summary.max_ms
                );
            }
            Ok::<_, tensor_store::ReaderError>(())
        })?;
    }

    Ok(())
}

#[cfg(target_os = "linux")]
fn tokio_prewarmed(_config: &ProfileConfig) -> ProfileResult {
    Err(ProfileError::new(
        "tokio safetensors cases are only compiled on non-Linux targets in this harness",
    )
    .into())
}

fn sync_load(config: &ProfileConfig) -> ProfileResult {
    use std::time::Instant;
    let fixtures = fixtures(config)?;

    for (fixture, files) in fixtures {
        let iterations = config.normalized_iterations();
        let cache_label = if config.cold_cache { "cold" } else { "warm" };
        let mut durations = Vec::with_capacity(iterations);
        let total_bytes = total_file_bytes(&files)?;
        println!(
            "Running sync safetensors load for '{}' ({iterations}x {cache_label})",
            fixture
        );
        for i in 0..iterations {
            if config.cold_cache && i == 0 {
                drop_page_cache_for_files(&files);
            }
            let start = Instant::now();
            let mut tensor_count = 0usize;
            for path in &files {
                let data = safetensors::Model::load_sync(path)?;
                tensor_count += data.names().len();
            }
            let elapsed = start.elapsed();
            durations.push(elapsed);
            println!(
                "  iteration {}: {} tensors, {} bytes, {:.2}ms",
                i + 1,
                tensor_count,
                total_bytes,
                elapsed.as_secs_f64() * 1000.0
            );
            black_box((total_bytes, tensor_count));
        }

        if let Some(summary) = summarize(&durations) {
            println!(
                "  summary: mean {:.2}ms | min {:.2}ms | max {:.2}ms",
                summary.mean_ms, summary.min_ms, summary.max_ms
            );
        }
    }

    Ok(())
}

fn mmap_load(config: &ProfileConfig) -> ProfileResult {
    use std::time::Instant;
    let fixtures = fixtures(config)?;

    for (fixture, files) in fixtures {
        let iterations = config.normalized_iterations();
        let cache_label = if config.cold_cache { "cold" } else { "warm" };
        let mut durations = Vec::with_capacity(iterations);
        let total_bytes = total_file_bytes(&files)?;
        println!(
            "Running mmap safetensors load for '{}' ({iterations}x {cache_label})",
            fixture
        );
        for i in 0..iterations {
            if config.cold_cache && i == 0 {
                drop_page_cache_for_files(&files);
            }
            let start = Instant::now();
            let mut tensor_count = 0usize;
            for path in &files {
                let data = safetensors::MmapModel::load(path)?;
                tensor_count += data.tensors().names().len();
            }
            let elapsed = start.elapsed();
            durations.push(elapsed);
            println!(
                "  iteration {}: {} tensors, {} bytes, {:.2}ms",
                i + 1,
                tensor_count,
                total_bytes,
                elapsed.as_secs_f64() * 1000.0
            );
            black_box((total_bytes, tensor_count));
        }

        if let Some(summary) = summarize(&durations) {
            println!(
                "  summary: mean {:.2}ms | min {:.2}ms | max {:.2}ms",
                summary.mean_ms, summary.min_ms, summary.max_ms
            );
        }
    }

    Ok(())
}

fn original_load(config: &ProfileConfig) -> ProfileResult {
    use std::time::Instant;
    let fixtures = fixtures(config)?;

    for (fixture, files) in fixtures {
        let iterations = config.normalized_iterations();
        let cache_label = if config.cold_cache { "cold" } else { "warm" };
        let mut durations = Vec::with_capacity(iterations);
        let total_bytes = total_file_bytes(&files)?;
        println!(
            "Running original safetensors load for '{}' ({iterations}x {cache_label})",
            fixture
        );
        for i in 0..iterations {
            if config.cold_cache && i == 0 {
                drop_page_cache_for_files(&files);
            }
            let start = Instant::now();
            let mut tensor_count = 0usize;
            for path in &files {
                let shard_bytes = fs::read(path)?;
                let data = SafeTensors::deserialize(&shard_bytes)?;
                tensor_count += data.names().len();
            }
            let elapsed = start.elapsed();
            durations.push(elapsed);
            println!(
                "  iteration {}: {} tensors, {} bytes, {:.2}ms",
                i + 1,
                tensor_count,
                total_bytes,
                elapsed.as_secs_f64() * 1000.0
            );
            black_box((tensor_count, total_bytes));
        }

        if let Some(summary) = summarize(&durations) {
            println!(
                "  summary: mean {:.2}ms | min {:.2}ms | max {:.2}ms",
                summary.mean_ms, summary.min_ms, summary.max_ms
            );
        }
    }

    Ok(())
}
