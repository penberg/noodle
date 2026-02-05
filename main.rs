use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

mod chat;

#[derive(Parser)]
#[command(name = "noodle")]
#[command(about = "A small language model", long_about = None)]
struct Opts {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum Backend {
    #[default]
    Gpu,
    Cuda,
    Cpu,
}

#[derive(Subcommand)]
enum Cmd {
    /// Train a model on a text corpus
    Train {
        /// Input text file
        input: PathBuf,

        /// Output directory for model files
        output: PathBuf,

        /// Backend to use for training
        #[arg(long, default_value = "gpu")]
        backend: Backend,

        /// Maximum number of training epochs
        #[arg(long, default_value = "1000")]
        max_epochs: usize,
    },

    /// Evaluate model perplexity on a test corpus
    Eval {
        /// Model file path
        model: PathBuf,

        /// Test corpus file
        corpus: PathBuf,

        /// Backend to use for evaluation
        #[arg(long, default_value = "gpu")]
        backend: Backend,
    },

    /// Chat with a trained model
    Chat {
        /// Model file
        model: PathBuf,

        /// Backend to use for inference
        #[arg(long, default_value = "gpu")]
        backend: Backend,
    },
}

fn main() {
    env_logger::init();
    let opts = Opts::parse();

    match opts.command {
        Cmd::Train {
            input,
            output,
            backend,
            max_epochs,
        } => {
            let backend = match backend {
                Backend::Gpu => noodle::Backend::Wgpu,
                Backend::Cuda => noodle::Backend::Cuda,
                Backend::Cpu => noodle::Backend::Cpu,
            };
            noodle::train(&input, &output, backend, max_epochs).expect("training failed");
        }
        Cmd::Eval {
            model,
            corpus,
            backend,
        } => {
            let backend = match backend {
                Backend::Gpu => noodle::Backend::Wgpu,
                Backend::Cuda => noodle::Backend::Cuda,
                Backend::Cpu => noodle::Backend::Cpu,
            };
            noodle::eval(&model, &corpus, backend).expect("evaluation failed");
        }
        Cmd::Chat { model, backend } => {
            let backend = match backend {
                Backend::Gpu => noodle::Backend::Wgpu,
                Backend::Cuda => noodle::Backend::Cuda,
                Backend::Cpu => noodle::Backend::Cpu,
            };
            chat::chat(&model, backend).expect("chat failed");
        }
    }
}
