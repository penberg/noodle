use argh::FromArgs;
use noodle::CorpusSource;
use std::path::PathBuf;

fn parse_corpus(value: &str) -> Result<CorpusSource, String> {
    CorpusSource::parse(value).map_err(|e| e.to_string())
}

/// A small language model
#[derive(FromArgs)]
pub struct Opts {
    #[argh(subcommand)]
    pub command: Cmd,
}

/// Wrapper around [`noodle::Backend`] so argh can parse it: the trait and the type
/// both live outside this crate, so the impl needs a local type to hang off.
#[derive(Clone, Copy, Debug, Default)]
pub struct Backend(noodle::Backend);

impl Backend {
    /// Backend to use when `--backend` is absent: whatever `NOODLE_BACKEND` says,
    /// or the library default.
    pub fn from_env() -> Self {
        Backend(noodle::Backend::from_env())
    }
}

impl argh::FromArgValue for Backend {
    fn from_arg_value(value: &str) -> Result<Self, String> {
        value.parse().map(Backend)
    }
}

impl From<Backend> for noodle::Backend {
    fn from(backend: Backend) -> Self {
        backend.0
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
    /// input text file, URL, or hf://owner/repo/file (streamed from HuggingFace)
    #[argh(positional, from_str_fn(parse_corpus))]
    pub input: CorpusSource,

    /// output directory for model files
    #[argh(positional)]
    pub output: PathBuf,

    /// backend to use for training (gpu, cuda, cpu)
    #[argh(option, default = "Backend::from_env()")]
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

    /// test corpus file, URL, or hf://owner/repo/file (streamed from HuggingFace)
    #[argh(positional, from_str_fn(parse_corpus))]
    pub corpus: CorpusSource,

    /// backend to use for evaluation (gpu, cuda, cpu)
    #[argh(option, default = "Backend::from_env()")]
    pub backend: Backend,
}

/// Fine-tune a pre-trained model on instruction data
#[derive(FromArgs)]
#[argh(subcommand, name = "finetune")]
pub struct FinetuneCmd {
    /// base model .mpk file path
    #[argh(positional)]
    pub model: PathBuf,

    /// input instruction text file, URL, or hf://owner/repo/file (streamed from HuggingFace)
    #[argh(positional, from_str_fn(parse_corpus))]
    pub input: CorpusSource,

    /// output directory for fine-tuned model
    #[argh(positional)]
    pub output: PathBuf,

    /// backend to use for training (gpu, cuda, cpu)
    #[argh(option, default = "Backend::from_env()")]
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
    #[argh(option, default = "Backend::from_env()")]
    pub backend: Backend,
}
