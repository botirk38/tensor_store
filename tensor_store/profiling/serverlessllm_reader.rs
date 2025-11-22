//! ServerlessLLM Reader Profiling Harness
//!
//! This profiling harness provides detailed performance analysis for serverlessllm reader operations.
//! Use with profiling tools like flamegraph, perf, or valgrind for deep performance insights.
//!
//! ## Usage Examples
//!
//! ### Flamegraph Profiling
//! ```bash
//! cargo flamegraph --bin serverlessllm_reader -- --profile async
//! cargo flamegraph --bin serverlessllm_reader -- --profile sync
//! ```
//!
//! ### Perf Profiling
//! ```bash
//! perf record -g cargo run --release --bin serverlessllm_reader -- --profile async
//! perf report
//! ```
//!
//! ### Valgrind Profiling
//! ```bash
//! valgrind --tool=callgrind cargo run --release --bin serverlessllm_reader -- --profile sync
//! kcachegrind callgrind.out.*
//! ```
//!
//! ## Test Data
//!
//! Requires a `test_model_serverlessllm/` directory containing:
//! - `tensor_index.json` - ServerlessLLM index file
//! - `tensor.data_0`, `tensor.data_1`, ... - Partitioned tensor data files
//!   For realistic profiling, use a large model with multiple data files.

use std::hint::black_box;
use std::time::Instant;
use tensor_store::readers::serverlessllm;
use tensor_store::readers::traits::TensorMetadata;

#[derive(Debug)]
enum ProfileMode {
    Async,
    Sync,
}

impl ProfileMode {
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "async" => Ok(Self::Async),
            "sync" => Ok(Self::Sync),
            _ => Err(format!("Unknown profile mode: {}. Use async or sync", s)),
        }
    }
}

async fn profile_async(serverlessllm_dir: &str) {
    println!("🔥 Profiling async serverlessllm reader...");

    // Warmup phase - populate buffer pools and caches
    println!("  Warming up...");
    for i in 0..3 {
        let _warmup = serverlessllm::load(serverlessllm_dir).await.unwrap();
        println!("    Warmup iteration {} complete", i + 1);
    }

    // Profiling phase - single iteration to avoid memory accumulation
    println!("  Starting profiling...");
    let start = Instant::now();
    let model = serverlessllm::load(black_box(serverlessllm_dir))
        .await
        .unwrap();
    let load_time = start.elapsed();

    let tensor_count = model.len();
    let mut bytes = 0;
    for (_name, tensor) in &model {
        bytes += tensor.data().len();
    }

    let avg_time = load_time;
    let avg_tensors = tensor_count;
    let avg_bytes = bytes;

    println!("\n📊 Async Profiling Results:");
    println!("  Average load time: {:.2}ms", avg_time.as_millis());
    println!("  Average tensors: {}", avg_tensors);
    println!("  Average bytes: {}", avg_bytes);
    println!("  Total profiling time: {:.2}s", load_time.as_secs_f64());
}

fn profile_sync(serverlessllm_dir: &str) {
    println!("🔥 Profiling sync serverlessllm reader...");

    // Warmup phase
    println!("  Warming up...");
    for i in 0..3 {
        let _warmup = serverlessllm::load_sync(serverlessllm_dir).unwrap();
        println!("    Warmup iteration {} complete", i + 1);
    }

    // Profiling phase
    println!("  Starting profiling...");
    let start = Instant::now();
    let model = serverlessllm::load_sync(black_box(serverlessllm_dir)).unwrap();
    let load_time = start.elapsed();

    let tensor_count = model.len();
    let mut bytes = 0;
    for (_name, tensor) in &model {
        bytes += tensor.data().len();
    }

    let avg_time = load_time;
    let avg_tensors = tensor_count;
    let avg_bytes = bytes;

    println!(
        "    Single iteration: {} tensors, {} bytes in {:.2}ms",
        tensor_count,
        bytes,
        load_time.as_millis()
    );

    println!("\n📊 Sync Profiling Results:");
    println!("  Load time: {:.2}ms", avg_time.as_millis());
    println!("  Tensors: {}", avg_tensors);
    println!("  Bytes: {}", avg_bytes);
    println!("  Total profiling time: {:.2}s", load_time.as_secs_f64());
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 3 || args[1] != "--profile" {
        eprintln!("Usage: {} --profile <mode>", args[0]);
        eprintln!("Modes: async, sync");
        eprintln!("\nRequires test_model_serverlessllm/ directory with:");
        eprintln!("  - tensor_index.json");
        eprintln!("  - tensor.data_0, tensor.data_1, ...");
        std::process::exit(1);
    }

    let mode = ProfileMode::from_str(&args[2]).map_err(|e| {
        eprintln!("{}", e);
        std::process::exit(1);
    })?;

    let serverlessllm_dir = "test_model_serverlessllm";

    // Check if test directory exists
    if !std::path::Path::new(serverlessllm_dir).exists() {
        eprintln!(
            "❌ Test directory '{}' not found in current directory",
            serverlessllm_dir
        );
        eprintln!("Please ensure test_model_serverlessllm/ exists for profiling");
        eprintln!("Directory should contain tensor_index.json and tensor.data_* files");
        std::process::exit(1);
    }

    println!("🚀 Starting serverlessllm reader profiling");
    println!("  Mode: {:?}", mode);
    println!("  Test directory: {}", serverlessllm_dir);
    println!();

    // Use minimal queue size for profiling to avoid memory conflicts with perf/flamegraph
    let mut binding = tokio_uring::builder();
    let builder = binding.entries(8);
    match builder.start(async {
        match mode {
            ProfileMode::Async => {
                profile_async(serverlessllm_dir).await;
            }
            ProfileMode::Sync => {
                profile_sync(serverlessllm_dir);
            }
        }

        println!("\n✅ Profiling complete!");
        Ok::<(), std::io::Error>(())
    }) {
        Ok(()) => Ok(()),
        Err(e) => {
            eprintln!("❌ Failed to initialize io_uring runtime: {}", e);
            eprintln!();
            eprintln!("💡 Troubleshooting:");
            eprintln!("  • If using flamegraph/perf: Try running without profiling tools first");
            eprintln!("  • If running as non-root: perf may be limited by memory locks");
            eprintln!(
                "  • Try increasing perf memory limits: sudo sysctl kernel.perf_event_mlock_kb=2048"
            );
            eprintln!(
                "  • Or run with: CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph --bin serverlessllm_reader -- --profile sync"
            );
            std::process::exit(1);
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 3 || args[1] != "--profile" {
        eprintln!("Usage: {} --profile <mode>", args[0]);
        eprintln!("Modes: async, sync");
        eprintln!("\nRequires test_model_serverlessllm/ directory with:");
        eprintln!("  - tensor_index.json");
        eprintln!("  - tensor.data_0, tensor.data_1, ...");
        std::process::exit(1);
    }

    let mode = ProfileMode::from_str(&args[2]).map_err(|e| {
        eprintln!("{}", e);
        std::process::exit(1);
    })?;

    let serverlessllm_dir = "test_model_serverlessllm";

    // Check if test directory exists
    if !std::path::Path::new(serverlessllm_dir).exists() {
        eprintln!(
            "❌ Test directory '{}' not found in current directory",
            serverlessllm_dir
        );
        eprintln!("Please ensure test_model_serverlessllm/ exists for profiling");
        eprintln!("Directory should contain tensor_index.json and tensor.data_* files");
        std::process::exit(1);
    }

    println!("🚀 Starting serverlessllm reader profiling");
    println!("  Mode: {:?}", mode);
    println!("  Test directory: {}", serverlessllm_dir);
    println!();

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        match mode {
            ProfileMode::Async => {
                profile_async(serverlessllm_dir).await;
            }
            ProfileMode::Sync => {
                profile_sync(serverlessllm_dir);
            }
        }

        println!("\n✅ Profiling complete!");
    });

    Ok(())
}
