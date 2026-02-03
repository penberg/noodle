#![recursion_limit = "256"]

pub mod inference;
pub mod model;
pub mod tokenizer;
pub mod train;

pub use inference::generate_next_token;
pub use tokenizer::{Token, Tokenize, decode};
pub use train::train;

#[derive(Clone, Copy, Debug, Default)]
pub enum Backend {
    #[default]
    Wgpu,
    Cuda,
    Cpu,
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
