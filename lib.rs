#![recursion_limit = "256"]

pub mod eval;
pub mod finetune;
pub mod inference;
pub mod model;
pub mod tokenizer;
pub mod train;

pub use eval::eval;
pub use finetune::finetune;
pub use inference::Session;
pub use tokenizer::{Token, Tokenizer};
pub use train::train;

/// Environment variable that selects the backend when nothing else does.
///
/// The `noodle` binary reads it as the fallback for `--backend`, and the benchmarks
/// read it as their only knob, so one variable picks the backend everywhere.
pub const BACKEND_ENV: &str = "NOODLE_BACKEND";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backend {
    #[default]
    Wgpu,
    Cuda,
    Cpu,
}

impl Backend {
    /// Short name for the backend, used in log lines and benchmark ids.
    pub fn name(self) -> &'static str {
        match self {
            Backend::Wgpu => "wgpu",
            Backend::Cuda => "cuda",
            Backend::Cpu => "cpu",
        }
    }

    /// Backend named by [`BACKEND_ENV`], or the default if the variable is unset.
    ///
    /// Panics on an unknown name rather than silently falling back: a typo that
    /// quietly ran on the wrong backend would make timings and logs misleading.
    pub fn from_env() -> Self {
        match std::env::var(BACKEND_ENV) {
            Ok(value) => value
                .parse()
                .unwrap_or_else(|e| panic!("{BACKEND_ENV}: {e}")),
            Err(_) => Backend::default(),
        }
    }
}

impl std::str::FromStr for Backend {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.to_lowercase().as_str() {
            "gpu" | "wgpu" => Ok(Backend::Wgpu),
            "cuda" => Ok(Backend::Cuda),
            "cpu" | "ndarray" => Ok(Backend::Cpu),
            other => Err(format!(
                "unknown backend '{other}' (expected one of: gpu, cuda, cpu)"
            )),
        }
    }
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Tokenizer(String),
    Burn(String),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "{}", e),
            Error::Tokenizer(s) => write!(f, "{}", s),
            Error::Burn(s) => write!(f, "{}", s),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
