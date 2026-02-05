use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use tiktoken_rs::p50k_base;

use crate::Result;

pub type Token = u32;

pub trait Tokenize {
    fn tokenize(&self) -> Result<Vec<Token>>;
}

impl Tokenize for str {
    fn tokenize(&self) -> Result<Vec<Token>> {
        let bpe = p50k_base().map_err(|e| crate::Error::Tokenizer(e.to_string()))?;
        let tokens = bpe.encode_with_special_tokens(self).into_iter().collect();
        Ok(tokens)
    }
}

impl Tokenize for Path {
    fn tokenize(&self) -> Result<Vec<Token>> {
        let file = File::open(self)?;
        let reader = BufReader::new(file);
        let bpe = p50k_base().map_err(|e| crate::Error::Tokenizer(e.to_string()))?;

        let mut tokens = Vec::new();
        let mut lines_processed = 0;

        for line in reader.lines() {
            let line = line?;
            let line_tokens: Vec<Token> =
                bpe.encode_with_special_tokens(&line).into_iter().collect();
            tokens.extend(line_tokens);

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

pub fn decode(tokens: &[Token]) -> Result<String> {
    let bpe = p50k_base().map_err(|e| crate::Error::Tokenizer(e.to_string()))?;
    bpe.decode(tokens.to_vec())
        .map_err(|e| crate::Error::Tokenizer(e.to_string()))
}
