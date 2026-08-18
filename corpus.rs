use std::{fs::File, io::Read, path::PathBuf};

use crate::Result;

/// A text corpus source: a local file, or a file streamed over HTTP —
/// e.g. directly from a HuggingFace dataset repository.
#[derive(Clone, Debug)]
pub enum CorpusSource {
    File(PathBuf),
    Url(String),
}

impl CorpusSource {
    /// Parse a corpus spec:
    ///
    /// - `hf://owner/repo/path/to/file` streams a file from a HuggingFace
    ///   dataset repository (via `huggingface.co/datasets/.../resolve/main/...`)
    /// - `http://` and `https://` URLs are streamed as-is
    /// - anything else is treated as a local file path
    pub fn parse(spec: &str) -> Result<Self> {
        if let Some(rest) = spec.strip_prefix("hf://") {
            let mut parts = rest.splitn(3, '/');
            match (parts.next(), parts.next(), parts.next()) {
                (Some(owner), Some(repo), Some(path))
                    if !owner.is_empty() && !repo.is_empty() && !path.is_empty() =>
                {
                    Ok(CorpusSource::Url(format!(
                        "https://huggingface.co/datasets/{owner}/{repo}/resolve/main/{path}"
                    )))
                }
                _ => Err(crate::Error::Corpus(format!(
                    "invalid HuggingFace dataset spec '{spec}' (expected hf://owner/repo/path/to/file)"
                ))),
            }
        } else if spec.starts_with("http://") || spec.starts_with("https://") {
            Ok(CorpusSource::Url(spec.to_string()))
        } else {
            Ok(CorpusSource::File(PathBuf::from(spec)))
        }
    }

    /// Open the corpus for reading. URLs are streamed over HTTP as data
    /// arrives; nothing is written to disk.
    pub fn open(&self) -> Result<Box<dyn Read>> {
        match self {
            CorpusSource::File(path) => Ok(Box::new(File::open(path)?)),
            CorpusSource::Url(url) => {
                let response = ureq::get(url)
                    .call()
                    .map_err(|e| crate::Error::Corpus(format!("failed to fetch {url}: {e}")))?;
                Ok(Box::new(response.into_reader()))
            }
        }
    }

    /// Read the entire corpus into a string.
    pub fn read_to_string(&self) -> Result<String> {
        let mut text = String::new();
        self.open()?.read_to_string(&mut text)?;
        Ok(text)
    }
}

impl std::fmt::Display for CorpusSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CorpusSource::File(path) => write!(f, "{}", path.display()),
            CorpusSource::Url(url) => write!(f, "{}", url),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hf_spec() {
        let source =
            CorpusSource::parse("hf://roneneldan/TinyStories/TinyStoriesV2-GPT4-train.txt")
                .unwrap();
        match source {
            CorpusSource::Url(url) => assert_eq!(
                url,
                "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
            ),
            other => panic!("expected Url, got {:?}", other),
        }
    }

    #[test]
    fn parse_hf_spec_with_nested_path() {
        let source = CorpusSource::parse("hf://owner/repo/data/shard-00.txt").unwrap();
        match source {
            CorpusSource::Url(url) => assert_eq!(
                url,
                "https://huggingface.co/datasets/owner/repo/resolve/main/data/shard-00.txt"
            ),
            other => panic!("expected Url, got {:?}", other),
        }
    }

    #[test]
    fn parse_hf_spec_without_file_is_rejected() {
        assert!(CorpusSource::parse("hf://roneneldan/TinyStories").is_err());
    }

    #[test]
    fn parse_url() {
        let source = CorpusSource::parse("https://example.com/corpus.txt").unwrap();
        assert!(matches!(source, CorpusSource::Url(_)));
    }

    #[test]
    fn parse_local_path() {
        let source = CorpusSource::parse("corpus/tinystories-train.txt").unwrap();
        assert!(matches!(source, CorpusSource::File(_)));
    }
}
