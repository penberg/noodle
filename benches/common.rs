//! Shared setup for the Noodle benchmarks.
//!
//! Both benchmark suites are generic over the burn backend and pick one at run time
//! from the `NOODLE_BACKEND` environment variable — the same variable the `noodle`
//! binary falls back to when `--backend` is absent:
//!
//! ```console
//! > cargo bench --bench inference                      # wgpu, the default
//! > NOODLE_BACKEND=cuda cargo bench --bench training   # CUDA
//! > NOODLE_BACKEND=cpu cargo bench --bench inference   # CPU (ndarray)
//! ```
#![allow(dead_code)]

use noodle::model::ModelConfig;

/// Which backend the benchmark runs on.
///
/// This is [`noodle::Backend`] itself, so the benchmarks accept exactly the names the
/// `--backend` option does. wgpu is the default for the same reason it is the default
/// there: it's the backend Noodle actually trains and generates on, so it's the one
/// whose timings mean something. CPU is available for machines without a usable GPU
/// adapter, but ndarray is several times slower and exercises a path nobody runs in
/// practice.
pub use noodle::Backend as BenchBackend;

/// Context length and vocabulary of the benchmarked model, spelled out as consts
/// because the benchmarks size their inputs from them.
///
/// They match the Sōmen preset (the default model `train.rs` builds: 4 layers,
/// emb_dim=256, 4 heads, 256-token context, p50k_base vocabulary), so the
/// measurements reflect the shapes the real training and inference paths run;
/// `bench_config` asserts they stay in sync.
pub const CTX_LEN: usize = 256;
pub const VOCAB_SIZE: usize = 50281;

/// Batch size used by pretraining (`train.rs`).
pub const TRAIN_BATCH_SIZE: usize = 8;

pub fn bench_config() -> ModelConfig {
    let config = ModelConfig::somen();
    assert_eq!(config.ctx_len, CTX_LEN);
    assert_eq!(config.vocab_size, VOCAB_SIZE);
    config
}

/// Deterministic pseudo-random token ids in `[0, VOCAB_SIZE)`.
///
/// Benchmarks need stable inputs across runs so timings are comparable, and an
/// untrained model's speed doesn't depend on the token values — only on the shapes.
/// A tiny xorshift keeps this reproducible without pulling the RNG's own cost into
/// the measured region.
pub fn fake_tokens(count: usize) -> Vec<u32> {
    let mut state: u32 = 0x9E37_79B9;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state % VOCAB_SIZE as u32
        })
        .collect()
}
