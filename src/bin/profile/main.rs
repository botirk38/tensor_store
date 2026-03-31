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
    },
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum SafeTensorsCase {
    /// Heuristic default load
    Default,
    /// Synchronous load
    Sync,
    /// Asynchronous load
    Async,
    /// Memory-mapped open
    Mmap,
    /// Explicit io_uring backend (Linux only)
    #[cfg(target_os = "linux")]
    IoUring,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ServerlessLLMCase {
    /// Heuristic default load
    Default,
    /// Synchronous load
    Sync,
    /// Asynchronous load
    Async,
    /// Memory-mapped open
    Mmap,
    /// Explicit io_uring backend (Linux only)
    #[cfg(target_os = "linux")]
    IoUring,
}

impl SafeTensorsCase {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Sync => "sync",
            Self::Async => "async",
            Self::Mmap => "mmap",
            #[cfg(target_os = "linux")]
            Self::IoUring => "io-uring",
        }
    }
}

impl ServerlessLLMCase {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Sync => "sync",
            Self::Async => "async",
            Self::Mmap => "mmap",
            #[cfg(target_os = "linux")]
            Self::IoUring => "io-uring",
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
        } => {
            let config = ProfileConfig {
                iterations,
                fixture,
            };
            safetensors::run(case.as_str(), &config)?;
        }
        Commands::Serverlessllm {
            case,
            fixture,
            iterations,
        } => {
            let config = ProfileConfig {
                iterations,
                fixture,
            };
            serverlessllm::run(case.as_str(), &config)?;
        }
    }

    Ok(())
}
