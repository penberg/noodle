use argh::FromArgs;
use std::path::PathBuf;

/// A small language model
#[derive(FromArgs)]
pub struct Opts {
    #[argh(subcommand)]
    pub command: Cmd,
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
pub enum Cmd {
    Train(TrainCmd),
    Eval(EvalCmd),
    Finetune(FinetuneCmd),
    Chat(ChatCmd),
}

/// Train a model on a text corpus
#[derive(FromArgs)]
#[argh(subcommand, name = "train")]
pub struct TrainCmd {
    /// input text file
    #[argh(positional)]
    pub input: PathBuf,

    /// output directory for model files
    #[argh(positional)]
    pub output: PathBuf,

    /// backend to use for training (gpu, cuda, cpu)
    #[argh(option, default = "Backend::Gpu")]
    pub backend: Backend,

    /// maximum number of training epochs
    #[argh(option, default = "1000")]
    pub max_epochs: usize,
}

/// Evaluate model perplexity on a test corpus
#[derive(FromArgs)]
#[argh(subcommand, name = "eval")]
pub struct EvalCmd {
    /// model file path
    #[argh(positional)]
    pub model: PathBuf,

    /// test corpus file
    #[argh(positional)]
    pub corpus: PathBuf,

    /// backend to use for evaluation (gpu, cuda, cpu)
    #[argh(option, default = "Backend::Gpu")]
    pub backend: Backend,
}

/// Fine-tune a pre-trained model on instruction data
#[derive(FromArgs)]
#[argh(subcommand, name = "finetune")]
pub struct FinetuneCmd {
    /// base model .mpk file path
    #[argh(positional)]
    pub model: PathBuf,

    /// input instruction text file
    #[argh(positional)]
    pub input: PathBuf,

    /// output directory for fine-tuned model
    #[argh(positional)]
    pub output: PathBuf,

    /// backend to use for training (gpu, cuda, cpu)
    #[argh(option, default = "Backend::Gpu")]
    pub backend: Backend,

    /// maximum number of fine-tuning epochs
    #[argh(option, default = "5")]
    pub max_epochs: usize,
}

/// Chat with a trained model
#[derive(FromArgs)]
#[argh(subcommand, name = "chat")]
pub struct ChatCmd {
    /// model file
    #[argh(positional)]
    pub model: PathBuf,

    /// backend to use for inference (gpu, cuda, cpu)
    #[argh(option, default = "Backend::Gpu")]
    pub backend: Backend,
}
