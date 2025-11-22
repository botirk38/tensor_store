//! Safetensors Reader Profiling Harness
//!
//! This profiling harness provides detailed performance analysis for safetensors reader operations.
//! Use with profiling tools like flamegraph, perf, or valgrind for deep performance insights.
//!
//! ## Usage Examples
//!
//! ### Flamegraph Profiling
//! ```bash
//! cargo flamegraph --bin safetensors_reader -- --profile async
//! cargo flamegraph --bin safetensors_reader -- --profile sync
//! ```
//!
//! ### Perf Profiling
//! ```bash
//! perf record -g cargo run --release --bin safetensors_reader -- --profile async
//! perf report
//! ```
//!
//! ### Valgrind Profiling
//! ```bash
//! valgrind --tool=callgrind cargo run --release --bin safetensors_reader -- --profile sync
//! kcachegrind callgrind.out.*
//! ```
//!
//! ## Test Data
//!
//! Requires a `test_model.safetensors` file in the current directory.
//! For realistic profiling, use a large model file (several GB).

use std::hint::black_box;
use std::time::Instant;
use tensor_store::readers::safetensors;

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

async fn profile_async(test_file: &str) {
    println!("🔥 Profiling async safetensors reader...");

    // Warmup phase - populate buffer pools and caches
    println!("  Warming up...");
    for i in 0..3 {
        let _warmup = safetensors::load(test_file).await.unwrap();
        println!("    Warmup iteration {} complete", i + 1);
    }

    // Profiling phase - single iteration to avoid memory accumulation
    println!("  Starting profiling...");
    let start = Instant::now();
    let data = safetensors::load(black_box(test_file)).await.unwrap();
    let load_time = start.elapsed();

    let tensor_count = data.names().len();
    let mut bytes = 0;
    for name in data.names() {
        if let Ok(tensor) = data.tensor(name) {
            bytes += tensor.data().len();
        }
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

fn profile_sync(test_file: &str) {
    println!("🔥 Profiling sync safetensors reader...");

    // Warmup phase
    println!("  Warming up...");
    for i in 0..3 {
        let _warmup = safetensors::load_sync(test_file).unwrap();
        println!("    Warmup iteration {} complete", i + 1);
    }

    // Profiling phase
    println!("  Starting profiling...");
    let start = Instant::now();
    let data = safetensors::load_sync(black_box(test_file)).unwrap();
    let load_time = start.elapsed();

    let tensor_count = data.names().len();
    let mut bytes = 0;
    for name in data.names() {
        if let Ok(tensor) = data.tensor(name) {
            bytes += tensor.data().len();
        }
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
        eprintln!("\nRequires test_model.safetensors in current directory");
        std::process::exit(1);
    }

    let mode = ProfileMode::from_str(&args[2]).map_err(|e| {
        eprintln!("{}", e);
        std::process::exit(1);
    })?;

    let test_file = "test_model.safetensors";

    // Check if test file exists
    if !std::path::Path::new(test_file).exists() {
        eprintln!(
            "❌ Test file '{}' not found in current directory",
            test_file
        );
        eprintln!("Please ensure test_model.safetensors exists for profiling");
        std::process::exit(1);
    }

    println!("🚀 Starting safetensors reader profiling");
    println!("  Mode: {:?}", mode);
    println!("  Test file: {}", test_file);
    println!();

    // Use minimal queue size for profiling to avoid memory conflicts with perf/flamegraph
    match tokio_uring::start(async {
        match mode {
            ProfileMode::Async => {
                profile_async(test_file).await;
            }
            ProfileMode::Sync => {
                profile_sync(test_file);
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
                "  • Or run with: CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph --bin safetensors_reader -- --profile sync"
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
        eprintln!("\nRequires test_model.safetensors in current directory");
        std::process::exit(1);
    }

    let mode = ProfileMode::from_str(&args[2]).map_err(|e| {
        eprintln!("{}", e);
        std::process::exit(1);
    })?;

    let test_file = "test_model.safetensors";

    // Check if test file exists
    if !std::path::Path::new(test_file).exists() {
        eprintln!(
            "❌ Test file '{}' not found in current directory",
            test_file
        );
        eprintln!("Please ensure test_model.safetensors exists for profiling");
        std::process::exit(1);
    }

    println!("🚀 Starting safetensors reader profiling");
    println!("  Mode: {:?}", mode);
    println!("  Test file: {}", test_file);
    println!();

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        match mode {
            ProfileMode::Async => {
                profile_async(test_file).await;
            }
            ProfileMode::Sync => {
                profile_sync(test_file);
            }
        }

        println!("\n✅ Profiling complete!");
    });

    Ok(())
}
