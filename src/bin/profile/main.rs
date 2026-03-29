use clap::{Parser, Subcommand, ValueEnum};

mod config;
mod safetensors;
mod serverlessllm;
mod stats;

use config::ProfileConfig;

#[derive(Parser)]
#[command(name = "profile")]
#[command(about = "Profiling harness for tensor_store (no Criterion)", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Profile SafeTensors loader
    Safetensors {
        /// Profiling case to run
        #[arg(value_enum)]
        case: SafeTensorsCase,

        /// Fixture name (e.g., qwen2-0.5b, mistral-7b)
        #[arg(short, long)]
        fixture: Option<String>,

        /// Number of iterations to run (default: 1)
        #[arg(short, long, default_value_t = 1)]
        iterations: usize,

        /// Attempt cold-cache profiling before the first iteration
        #[arg(long, default_value_t = false)]
        cold_cache: bool,
    },
    /// Profile ServerlessLLM loader
    Serverlessllm {
        /// Profiling case to run
        #[arg(value_enum)]
        case: ServerlessLLMCase,

        /// Fixture name (e.g., qwen2-0.5b, mistral-7b)
        #[arg(short, long)]
        fixture: Option<String>,

        /// Number of iterations to run (default: 1)
        #[arg(short, long, default_value_t = 1)]
        iterations: usize,

        /// Attempt cold-cache profiling before the first iteration
        #[arg(long, default_value_t = false)]
        cold_cache: bool,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum SafeTensorsCase {
    /// io_uring async load (Linux only)
    IoUringLoad,
    /// io_uring prewarmed load (Linux only)
    IoUringPrewarmed,
    /// Tokio async load
    TokioLoad,
    /// Tokio prewarmed load
    TokioPrewarmed,
    /// Synchronous load
    Sync,
    /// Memory-mapped load
    Mmap,
    /// Original safetensors crate load
    Original,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ServerlessLLMCase {
    /// Async load
    Async,
    /// Synchronous load
    Sync,
    /// Memory-mapped load
    Mmap,
}

impl SafeTensorsCase {
    fn as_str(&self) -> &'static str {
        match self {
            Self::IoUringLoad => "io-uring-load",
            Self::IoUringPrewarmed => "io-uring-prewarmed",
            Self::TokioLoad => "tokio-load",
            Self::TokioPrewarmed => "tokio-prewarmed",
            Self::Sync => "sync",
            Self::Mmap => "mmap",
            Self::Original => "original",
        }
    }
}

impl ServerlessLLMCase {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Async => "async-load",
            Self::Sync => "sync-load",
            Self::Mmap => "mmap-load",
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Safetensors {
            case,
            fixture,
            iterations,
            cold_cache,
        } => {
            let config = ProfileConfig {
                iterations,
                fixture,
                cold_cache,
            };
            safetensors::run(case.as_str(), &config)?;
        }
        Commands::Serverlessllm {
            case,
            fixture,
            iterations,
            cold_cache,
        } => {
            let config = ProfileConfig {
                iterations,
                fixture,
                cold_cache,
            };
            serverlessllm::run(case.as_str(), &config)?;
        }
    }

    Ok(())
}
