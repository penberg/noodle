use argh::FromArgs;
use std::path::PathBuf;

mod chat;

/// A small language model
#[derive(FromArgs)]
struct Opts {
    #[argh(subcommand)]
    command: Cmd,
}

#[derive(Clone, Copy, Debug, Default)]
pub enum Backend {
    #[default]
    Gpu,
    Cuda,
    Cpu,
}

impl argh::FromArgValue for Backend {
    fn from_arg_value(value: &str) -> Result<Self, String> {
        match value {
            "gpu" => Ok(Backend::Gpu),
            "cuda" => Ok(Backend::Cuda),
            "cpu" => Ok(Backend::Cpu),
            _ => Err(format!(
                "unknown backend '{value}' (expected one of: gpu, cuda, cpu)"
            )),
        }
    }
}

impl From<Backend> for noodle::Backend {
    fn from(backend: Backend) -> Self {
        match backend {
            Backend::Gpu => noodle::Backend::Wgpu,
            Backend::Cuda => noodle::Backend::Cuda,
            Backend::Cpu => noodle::Backend::Cpu,
        }
    }
}

#[derive(FromArgs)]
#[argh(subcommand)]
enum Cmd {
    Train(TrainCmd),
    Eval(EvalCmd),
    Finetune(FinetuneCmd),
    Chat(ChatCmd),
}

/// Train a model on a text corpus
#[derive(FromArgs)]
#[argh(subcommand, name = "train")]
struct TrainCmd {
    /// input text file
    #[argh(positional)]
    input: PathBuf,

    /// output directory for model files
    #[argh(positional)]
    output: PathBuf,

    /// backend to use for training (gpu, cuda, cpu)
    #[argh(option, default = "Backend::Gpu")]
    backend: Backend,

    /// maximum number of training epochs
    #[argh(option, default = "1000")]
    max_epochs: usize,
}

/// Evaluate model perplexity on a test corpus
#[derive(FromArgs)]
#[argh(subcommand, name = "eval")]
struct EvalCmd {
    /// model file path
    #[argh(positional)]
    model: PathBuf,

    /// test corpus file
    #[argh(positional)]
    corpus: PathBuf,

    /// backend to use for evaluation (gpu, cuda, cpu)
    #[argh(option, default = "Backend::Gpu")]
    backend: Backend,
}

/// Fine-tune a pre-trained model on instruction data
#[derive(FromArgs)]
#[argh(subcommand, name = "finetune")]
struct FinetuneCmd {
    /// base model .mpk file path
    #[argh(positional)]
    model: PathBuf,

    /// input instruction text file
    #[argh(positional)]
    input: PathBuf,

    /// output directory for fine-tuned model
    #[argh(positional)]
    output: PathBuf,

    /// backend to use for training (gpu, cuda, cpu)
    #[argh(option, default = "Backend::Gpu")]
    backend: Backend,

    /// maximum number of fine-tuning epochs
    #[argh(option, default = "5")]
    max_epochs: usize,
}

/// Chat with a trained model
#[derive(FromArgs)]
#[argh(subcommand, name = "chat")]
struct ChatCmd {
    /// model file
    #[argh(positional)]
    model: PathBuf,

    /// backend to use for inference (gpu, cuda, cpu)
    #[argh(option, default = "Backend::Gpu")]
    backend: Backend,
}

fn main() {
    env_logger::init();
    let opts: Opts = argh::from_env();

    match opts.command {
        Cmd::Train(cmd) => {
            noodle::train(&cmd.input, &cmd.output, cmd.backend.into(), cmd.max_epochs)
                .expect("training failed");
        }
        Cmd::Finetune(cmd) => {
            noodle::finetune(
                &cmd.model,
                &cmd.input,
                &cmd.output,
                cmd.backend.into(),
                cmd.max_epochs,
            )
            .expect("fine-tuning failed");
        }
        Cmd::Eval(cmd) => {
            noodle::eval(&cmd.model, &cmd.corpus, cmd.backend.into()).expect("evaluation failed");
        }
        Cmd::Chat(cmd) => {
            chat::chat(&cmd.model, cmd.backend.into()).expect("chat failed");
        }
    }
}
