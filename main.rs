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
    let opts = Opts::parse();

    match opts.command {
        Cmd::Train {
            input,
            output,
            backend,
            max_epochs,
        } => {
            let use_gpu = matches!(backend, Backend::Gpu);
            noodle::train(&input, &output, use_gpu, max_epochs).expect("training failed");
        }
        Cmd::Chat { model, backend } => {
            let use_gpu = matches!(backend, Backend::Gpu);
            chat::chat(&model, use_gpu).expect("chat failed");
        }
    }
}
