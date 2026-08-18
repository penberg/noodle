#![recursion_limit = "256"]

pub mod corpus;
pub mod eval;
pub mod finetune;
pub mod inference;
pub mod model;
pub mod tokenizer;
pub mod train;

pub use corpus::CorpusSource;
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

/// Floating-point width for model compute, chosen at runtime from what the
/// device actually supports rather than hard-coded per backend: the same wgpu
/// build lands on different hardware, and NdArray simply has no half types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Bf16,
    F16,
    F32,
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Precision::Bf16 => write!(f, "bf16"),
            Precision::F16 => write!(f, "f16"),
            Precision::F32 => write!(f, "f32"),
        }
    }
}

/// Precision to train at on `device`: bf16 when the device supports it, f32
/// otherwise. Only bf16 is considered for the reduced option because it keeps
/// f32's exponent range, so training needs no loss scaling; f16 would.
pub fn training_precision<B: burn::prelude::Backend>(device: &B::Device) -> Precision {
    use burn::tensor::DType;

    if B::supports_dtype(device, DType::BF16) {
        Precision::Bf16
    } else {
        Precision::F32
    }
}

/// Precision to run inference at on `device`: the narrowest supported float,
/// probing bf16, then f16, then falling back to f32. Inference is a forward
/// pass only, so f16's small exponent range is acceptable where bf16 is not
/// available (WGSL has no bf16, but many adapters expose shader f16).
pub fn inference_precision<B: burn::prelude::Backend>(device: &B::Device) -> Precision {
    use burn::tensor::DType;

    if B::supports_dtype(device, DType::BF16) {
        Precision::Bf16
    } else if B::supports_dtype(device, DType::F16) {
        Precision::F16
    } else {
        Precision::F32
    }
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Tokenizer(String),
    Burn(String),
    Corpus(String),
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
            Error::Corpus(s) => write!(f, "{}", s),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
