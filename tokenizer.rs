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

    /// Decode tokens to raw bytes. A BPE token holds an arbitrary byte
    /// sequence, so the result may start or end in the middle of a multi-byte
    /// character; callers streaming token by token must buffer the bytes
    /// until they form valid UTF-8.
    pub fn decode_bytes(&self, tokens: &[Token]) -> Vec<u8> {
        self.bpe
            ._decode_native_and_split(tokens.to_vec())
            .flatten()
            .collect()
    }

    /// Decode tokens, substituting U+FFFD for invalid or incomplete UTF-8
    /// instead of failing. Sampled output can legitimately end mid-character.
    pub fn decode_lossy(&self, tokens: &[Token]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(tokens)).into_owned()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A multi-byte character can be split across tokens, so decoding a
    /// single token may not be valid UTF-8 — but the bytes always are once
    /// concatenated.
    #[test]
    fn decode_bytes_reassembles_split_characters() {
        let tokenizer = Tokenizer::new().unwrap();
        let text = "Sōmen — “noodles”";
        let tokens = tokenizer.encode(text);

        let per_token: Vec<u8> = tokens
            .iter()
            .flat_map(|&t| tokenizer.decode_bytes(&[t]))
            .collect();
        assert_eq!(per_token, text.as_bytes());

        // At least one token in this text ends mid-character, which is the
        // case that makes strict single-token decode fail.
        assert!(
            tokens
                .iter()
                .any(|&t| std::str::from_utf8(&tokenizer.decode_bytes(&[t])).is_err())
        );
    }

    #[test]
    fn decode_lossy_tolerates_partial_characters() {
        let tokenizer = Tokenizer::new().unwrap();
        let tokens = tokenizer.encode("Sōmen — “noodles”");
        let split = tokens
            .iter()
            .find(|&&t| std::str::from_utf8(&tokenizer.decode_bytes(&[t])).is_err())
            .expect("text should produce a token ending mid-character");

        assert!(tokenizer.decode(&[*split]).is_err());
        assert!(tokenizer.decode_lossy(&[*split]).contains('\u{FFFD}'));
    }
}
