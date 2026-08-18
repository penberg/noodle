use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};

use tiktoken_rs::{CoreBPE, p50k_base};

use crate::Result;

pub type Token = u32;

/// BPE tokenizer wrapping tiktoken's p50k_base encoding.
pub struct Tokenizer {
    bpe: CoreBPE,
}

impl Tokenizer {
    pub fn new() -> Result<Self> {
        let bpe = p50k_base().map_err(|e| crate::Error::Tokenizer(e.to_string()))?;
        Ok(Self { bpe })
    }

    pub fn encode(&self, text: &str) -> Vec<Token> {
        self.bpe
            .encode_with_special_tokens(text)
            .into_iter()
            .collect()
    }

    pub fn decode(&self, tokens: &[Token]) -> Result<String> {
        self.bpe
            .decode(tokens.to_vec())
            .map_err(|e| crate::Error::Tokenizer(e.to_string()))
    }

    /// Tokenize a file line-by-line, preserving newlines.
    pub fn encode_file(&self, path: &Path) -> Result<Vec<Token>> {
        self.encode_reader(File::open(path)?)
    }

    /// Tokenize a stream line-by-line, preserving newlines. Tokens are
    /// produced as data arrives, so the full text is never held in memory.
    pub fn encode_reader<R: Read>(&self, reader: R) -> Result<Vec<Token>> {
        let reader = BufReader::new(reader);

        let mut tokens = Vec::new();
        let mut lines_processed = 0;

        for line in reader.lines() {
            let line = line? + "\n";
            tokens.extend(self.encode(&line));

            lines_processed += 1;
            if lines_processed % 100_000 == 0 {
                eprintln!(
                    "  tokenized {} lines, {} tokens so far...",
                    lines_processed,
                    tokens.len()
                );
            }
        }

        eprintln!(
            "  tokenized {} lines, {} tokens total",
            lines_processed,
            tokens.len()
        );
        Ok(tokens)
    }
}
