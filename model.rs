//! GPT-style decoder-only transformer model.

use std::{fs, path::Path};

use burn::{
    grad_clipping::GradientClippingConfig,
    module::{AutodiffModule, Module},
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Initializer, LayerNorm,
        LayerNormConfig, Linear, LinearConfig, loss::CrossEntropyLossConfig,
    },
    optim::{AdamW, AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::Backend,
    tensor::{
        Bool, ElementConversion, Int, Tensor, TensorData, activation, backend::AutodiffBackend,
    },
};
use serde::{Deserialize, Serialize};

use crate::Result;

const DROPOUT_RATE: f64 = 0.1;

/// Vocabulary size of the p50k_base tokenizer, which every preset shares.
const P50K_VOCAB_SIZE: usize = 50281;

/// Configuration for the decoder-only transformer model architecture.
///
/// Field names follow the conventions of open-weight model configs (Qwen, Llama):
/// `n_layers`, `emb_dim`, `n_heads`, `head_dim`, `hidden_dim`. The named presets
/// ([`somen`](Self::somen), [`soba`](Self::soba), [`udon`](Self::udon)) record
/// their name in a saved `model.json` purely as information — loading derives
/// everything from the architecture numbers, so checkpoints stay self-describing.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(from = "ModelConfigCompat")]
pub struct ModelConfig {
    /// Which preset built this config, if any. Purely informative: nothing is
    /// derived from it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,
    pub n_layers: usize,
    pub emb_dim: usize,
    pub n_heads: usize,
    /// Width of each attention head, decoupled from `emb_dim / n_heads` (as in
    /// Qwen3): attention runs at `n_heads * head_dim` wide and projects back to
    /// `emb_dim`. Must be even for rotary position embeddings.
    pub head_dim: usize,
    /// Intermediate width of the feed-forward network.
    pub hidden_dim: usize,
    pub ctx_len: usize,
    pub vocab_size: usize,
}

/// Accepts both the current field names and the ones older checkpoints were saved
/// with (`layers`/`d_model`/`heads`, with no `head_dim` or `hidden_dim`), so
/// existing `model.json` files keep loading.
#[derive(Deserialize)]
struct ModelConfigCompat {
    #[serde(default)]
    preset: Option<String>,
    #[serde(alias = "layers")]
    n_layers: usize,
    #[serde(alias = "d_model")]
    emb_dim: usize,
    #[serde(alias = "heads")]
    n_heads: usize,
    head_dim: Option<usize>,
    hidden_dim: Option<usize>,
    ctx_len: usize,
    vocab_size: usize,
}

impl From<ModelConfigCompat> for ModelConfig {
    fn from(c: ModelConfigCompat) -> Self {
        // Older configs predate the decoupled head width and configurable FFN
        // width; they were built with head_dim = emb_dim / n_heads and
        // hidden_dim = 4 * emb_dim.
        Self {
            preset: c.preset,
            n_layers: c.n_layers,
            emb_dim: c.emb_dim,
            n_heads: c.n_heads,
            head_dim: c.head_dim.unwrap_or(c.emb_dim / c.n_heads),
            hidden_dim: c.hidden_dim.unwrap_or(4 * c.emb_dim),
            ctx_len: c.ctx_len,
            vocab_size: c.vocab_size,
        }
    }
}

impl ModelConfig {
    /// Sōmen (~29M parameters): the thinnest noodle, cooked in about ninety
    /// seconds. A smoke-test model for exercising the pipeline quickly, sized for
    /// small datasets (~300K-1M tokens) — not a real model.
    pub fn somen() -> Self {
        Self {
            preset: Some("Sōmen".to_string()),
            n_layers: 4,
            emb_dim: 256,
            n_heads: 4,
            head_dim: 64,
            hidden_dim: 1024,
            ctx_len: 256,
            vocab_size: P50K_VOCAB_SIZE,
        }
    }

    /// Soba (~0.6B parameters): the everyday noodle, for real training runs while
    /// iterating. Mirrors Qwen3-0.6B's shape (28 layers, emb_dim 1024, 16 heads of
    /// width 128), except hidden_dim is 4096 rather than Qwen3's 3072: our FFN is a
    /// two-matrix GELU rather than their three-matrix SwiGLU, so it needs a wider
    /// intermediate to spend the same parameter budget.
    pub fn soba() -> Self {
        Self {
            preset: Some("Soba".to_string()),
            n_layers: 28,
            emb_dim: 1024,
            n_heads: 16,
            head_dim: 128,
            hidden_dim: 4096,
            ctx_len: 1024,
            vocab_size: P50K_VOCAB_SIZE,
        }
    }

    /// Udon (~27B parameters): the thick noodle — the target model, in a
    /// Llama-33B-class shape. Training and running it locally still needs work the
    /// config alone can't provide (bf16, grouped-query attention, checkpointing).
    pub fn udon() -> Self {
        Self {
            preset: Some("Udon".to_string()),
            n_layers: 50,
            emb_dim: 6656,
            n_heads: 52,
            head_dim: 128,
            hidden_dim: 26624,
            ctx_len: 4096,
            vocab_size: P50K_VOCAB_SIZE,
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let config_path = path.with_extension("json");
        let config_json = fs::read_to_string(&config_path)?;
        serde_json::from_str(&config_json).map_err(|e| crate::Error::Burn(e.to_string()))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let config_path = path.with_extension("json");
        let config_json =
            serde_json::to_string_pretty(self).map_err(|e| crate::Error::Burn(e.to_string()))?;
        fs::write(&config_path, config_json)?;
        Ok(())
    }
}

/// The language model: a decoder-only transformer neural network.
///
/// This struct represents the core model architecture with token/position embeddings,
/// transformer blocks, and output projection. It implements the forward pass only.
///
/// Used for both inference and training:
/// - For inference, use `Model<B>` directly with any `Backend`
/// - For training, wrap it in a [`Trainer`] which adds optimizer and gradient support
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    token_emb: Embedding<B>,
    blocks: Vec<TransformerBlock<B>>,
    ln_f: LayerNorm<B>,
    output: Linear<B>,
    ctx_len: usize,
    vocab_size: usize,
    head_dim: usize,
}

impl<B: Backend> Model<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        eprintln!("  initializing...");

        // Rotary position embeddings rotate the head dimension in pairs, so head_dim
        // must be even. Fail early with a clear message rather than panicking on a
        // shape mismatch deep in the forward pass.
        assert!(
            config.head_dim.is_multiple_of(2),
            "head_dim ({}) must be even for rotary position embeddings",
            config.head_dim,
        );

        eprintln!(
            "  creating token embeddings ({} x {})...",
            config.vocab_size, config.emb_dim
        );
        let emb_init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        let token_emb = EmbeddingConfig::new(config.vocab_size, config.emb_dim)
            .with_initializer(emb_init)
            .init(device);

        eprintln!("  using rotary position embeddings (RoPE)");

        eprintln!("  creating {} transformer blocks...", config.n_layers);
        let mut blocks = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            let block = TransformerBlock::new(config, device);
            blocks.push(block);
        }

        eprintln!("  creating final layer norm...");
        let ln_f = LayerNormConfig::new(config.emb_dim).init(device);

        eprintln!("  creating output projection...");
        let init = Initializer::XavierUniform { gain: 1.0 };
        let output = LinearConfig::new(config.emb_dim, config.vocab_size)
            .with_initializer(init)
            .init(device);

        eprintln!("  model ready");

        Self {
            token_emb,
            blocks,
            ln_f,
            output,
            ctx_len: config.ctx_len,
            vocab_size: config.vocab_size,
            head_dim: config.head_dim,
        }
    }

    pub fn load(path: &Path, device: &B::Device) -> Result<Self> {
        let config = ModelConfig::load(path)?;

        match &config.preset {
            Some(name) => eprintln!(
                "Loading model: {name} ({} layers, emb_dim={})",
                config.n_layers, config.emb_dim
            ),
            None => eprintln!(
                "Loading model: {} layers, emb_dim={}",
                config.n_layers, config.emb_dim
            ),
        }

        let model = Self::new(&config, device);

        model
            .load_file(
                path,
                &burn::record::DefaultFileRecorder::<burn::record::FullPrecisionSettings>::new(),
                device,
            )
            .map_err(|e| crate::Error::Burn(e.to_string()))
    }

    pub fn ctx_len(&self) -> usize {
        self.ctx_len
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Forward pass: [batch, seq_len] -> [batch, seq_len, vocab_size]
    /// Creates position IDs and causal mask internally based on sequence length.
    pub fn forward(&self, token_ids: Tensor<B, 2, Int>, device: &B::Device) -> Tensor<B, 3> {
        let [_batch, seq_len] = token_ids.dims();

        // Rotary position embeddings for positions 0..seq_len, shared across all blocks.
        // Position information is injected by rotating Q and K inside attention rather
        // than by adding an absolute position embedding here.
        let (cos, sin) = rope_tables::<B>(0, seq_len, self.head_dim, device);
        let mask = causal_mask::<B>(seq_len, device);

        let mut x = self.token_emb.forward(token_ids);

        for block in &self.blocks {
            x = block.forward(x, &mask, &cos, &sin);
        }

        let x = self.ln_f.forward(x);

        self.output.forward(x)
    }

    /// An empty key/value cache sized for this model.
    pub fn new_cache(&self) -> KvCache<B> {
        KvCache::new(self.blocks.len())
    }

    /// Forward pass that reuses cached keys and values: [1, seq_len] -> [1, vocab_size].
    ///
    /// The tokens are treated as continuing whatever is already in `cache`, and their keys
    /// and values are appended to it. Only the logits for the final position are returned:
    /// generation samples from that position alone, and the output head is the most
    /// expensive matmul in the model, so the hidden state is sliced before the projection
    /// rather than after it. A linear layer is pointwise across positions, so that is the
    /// same answer for less work.
    ///
    /// Two shapes of call are supported:
    /// - **Prefill**, into an empty cache: any `seq_len`, masked causally.
    /// - **Decode**, continuing a non-empty cache: exactly one token, which needs no mask
    ///   at all, since a single query attends to the cache and to itself, all of which is
    ///   in the past.
    ///
    /// Prefilling on top of a non-empty cache would need a rectangular mask and is not
    /// supported; feed those tokens one at a time instead.
    pub fn forward_cached(
        &self,
        token_ids: Tensor<B, 2, Int>,
        cache: &mut KvCache<B>,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let [batch, seq_len] = token_ids.dims();
        assert_eq!(batch, 1, "the kv cache only supports a batch size of 1");

        let pos = cache.len();
        assert!(
            pos + seq_len <= self.ctx_len,
            "cached tokens ({pos}) plus new tokens ({seq_len}) exceed the context length ({})",
            self.ctx_len,
        );

        let mask = if pos == 0 {
            Some(causal_mask::<B>(seq_len, device))
        } else {
            assert_eq!(
                seq_len, 1,
                "only one token at a time can be appended to a non-empty cache",
            );
            None
        };

        // Rotate the new tokens at their absolute offset, so they sit at the right distance
        // from the keys already in the cache.
        let (cos, sin) = rope_tables::<B>(pos, seq_len, self.head_dim, device);

        let mut x = self.token_emb.forward(token_ids);

        for (block, layer) in self.blocks.iter().zip(cache.layers.iter_mut()) {
            x = block.forward_cached(x, mask.as_ref(), &cos, &sin, layer);
        }
        cache.len += seq_len;

        let x = self.ln_f.forward(x);
        let x = x.slice([0..1, (seq_len - 1)..seq_len]);
        let x = self.output.forward(x);

        x.reshape([1, self.vocab_size])
    }
}

/// The keys and values computed for every token the model has already seen, held per layer
/// so that generating a token costs one position of work instead of re-running the whole
/// context.
///
/// Entries are stored with their rotary embedding already applied at the absolute position
/// they were added at. That is what lets a caller evict from the front without rewriting
/// the cache: RoPE makes `q_p . k_j` depend only on `p - j`, so dropping old entries leaves
/// every surviving pair at the distance it had before.
#[derive(Debug)]
pub struct KvCache<B: Backend> {
    layers: Vec<LayerCache<B>>,
    len: usize,
}

impl<B: Backend> KvCache<B> {
    fn new(layers: usize) -> Self {
        Self {
            layers: (0..layers).map(|_| LayerCache::new()).collect(),
            len: 0,
        }
    }

    /// Number of tokens currently cached.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Drop everything, so the next call starts from position zero again.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.kv = None;
        }
        self.len = 0;
    }
}

/// One layer's slice of a [`KvCache`]: keys and values of shape `[1, n_heads, len, head_dim]`.
#[derive(Debug)]
pub struct LayerCache<B: Backend> {
    kv: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
}

impl<B: Backend> LayerCache<B> {
    fn new() -> Self {
        Self { kv: None }
    }
}

/// Build the additive causal attention mask: `[1, 1, seq_len, seq_len]`, holding 0 where a
/// position may attend and a large negative value where it may not.
fn causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 4> {
    // triu_mask returns FALSE for upper triangle (future), TRUE for lower triangle (past/current)
    // (it's a "mask for upper triangle operation", not "upper triangle is true")
    let attn_mask: Tensor<B, 2, Bool> = Tensor::triu_mask([seq_len, seq_len], 1, device);
    let zeros: Tensor<B, 2> = Tensor::zeros([seq_len, seq_len], device);
    let large_neg: Tensor<B, 2> = Tensor::full([seq_len, seq_len], -1e9f32, device);

    // mask_where: self.mask_where(mask, value) → value where TRUE, self where FALSE
    // TRUE (past/current, j ≤ i) → zeros (can attend)
    // FALSE (future, j > i) → large_neg (blocked)
    let mask = large_neg.mask_where(attn_mask, zeros);

    mask.reshape([1, 1, seq_len, seq_len])
}
/// Base frequency for rotary position embeddings (RoPE), following the original paper.
const ROPE_BASE: f32 = 10_000.0;

/// Build the RoPE cosine and sine tables for `len` positions starting at `pos`.
///
/// Rotary position embeddings encode absolute position by rotating pairs of dimensions
/// in the Q and K vectors by an angle proportional to the position. Dimension pair `i`
/// rotates at frequency `ROPE_BASE^(-2i/d_head)`, so low dimensions rotate slowly (coarse
/// position) and high dimensions rotate quickly (fine position). Because the rotation is a
/// linear map, the attention dot product `q·k` ends up depending only on the *relative*
/// distance between the two positions — the property that makes RoPE generalize across
/// sequence lengths without any learned position parameters.
///
/// The `pos` offset is what lets a decode step rotate a single token as position 200
/// rather than position 0: attention depends only on the distance between two positions,
/// so a cached key keeps the rotation it was built with, and a new query has to be built
/// at its true absolute position for that distance to come out right.
///
/// Returns `(cos, sin)`, each of shape `[1, 1, len, d_head]` so they broadcast over the
/// `[batch, heads, seq_len, d_head]` Q/K tensors. This uses the "rotate-half" layout
/// (as in LLaMA/GPT-NeoX): the frequency vector is duplicated so the first and second
/// halves of `d_head` share angles, pairing dimension `i` with dimension `i + d_head/2`.
fn rope_tables<B: Backend>(
    pos: usize,
    len: usize,
    d_head: usize,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let half = d_head / 2;

    // inv_freq[i] = ROPE_BASE^(-2i/d_head) for i in 0..half
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| ROPE_BASE.powf(-(2.0 * i as f32) / d_head as f32))
        .collect();
    let inv_freq = Tensor::<B, 1>::from_data(TensorData::new(inv_freq, [half]), device);

    let positions: Vec<f32> = (pos..pos + len).map(|p| p as f32).collect();
    let positions = Tensor::<B, 1>::from_data(TensorData::new(positions, [len]), device);

    // Outer product: freqs[p, i] = position p * inv_freq[i] -> [len, half]
    let freqs = positions.reshape([len, 1]) * inv_freq.reshape([1, half]);

    // Duplicate along the feature dim so angles line up with the rotate-half layout.
    let emb = Tensor::cat(vec![freqs.clone(), freqs], 1); // [len, d_head]

    let cos = emb.clone().cos().reshape([1, 1, len, d_head]);
    let sin = emb.sin().reshape([1, 1, len, d_head]);
    (cos, sin)
}

/// Apply rotary position embeddings to a Q or K tensor of shape `[batch, heads, seq_len, d_head]`.
///
/// Implements `x_rotated = x * cos + rotate_half(x) * sin`, where `rotate_half` splits the
/// last dimension into halves `[x1, x2]` and returns `[-x2, x1]`. This is the matrix form of
/// rotating each `(x_i, x_{i+d_head/2})` pair by its position-dependent angle.
fn apply_rope<B: Backend>(x: Tensor<B, 4>, cos: &Tensor<B, 4>, sin: &Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, heads, seq_len, d_head] = x.dims();
    let half = d_head / 2;

    let x1 = x.clone().slice([0..batch, 0..heads, 0..seq_len, 0..half]);
    let x2 = x
        .clone()
        .slice([0..batch, 0..heads, 0..seq_len, half..d_head]);
    let rotated = Tensor::cat(vec![x2.neg(), x1], 3);

    x * cos.clone() + rotated * sin.clone()
}

/// A single transformer block with multi-head self-attention and feed-forward network.
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    ln1: LayerNorm<B>,
    attn_qkv: Linear<B>,
    attn_proj: Linear<B>,
    attn_dropout: Dropout,
    ln2: LayerNorm<B>,
    ffn_up: Linear<B>,
    ffn_down: Linear<B>,
    ffn_dropout: Dropout,
    n_heads: usize,
    head_dim: usize,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        // Attention runs at n_heads * head_dim wide, which need not equal emb_dim:
        // the QKV projection widens into it and the output projection maps back.
        let attn_width = config.n_heads * config.head_dim;
        let init = Initializer::XavierUniform { gain: 1.0 };
        Self {
            ln1: LayerNormConfig::new(config.emb_dim).init(device),
            attn_qkv: LinearConfig::new(config.emb_dim, 3 * attn_width)
                .with_initializer(init.clone())
                .init(device),
            attn_proj: LinearConfig::new(attn_width, config.emb_dim)
                .with_initializer(init.clone())
                .init(device),
            attn_dropout: DropoutConfig::new(DROPOUT_RATE).init(),
            ln2: LayerNormConfig::new(config.emb_dim).init(device),
            ffn_up: LinearConfig::new(config.emb_dim, config.hidden_dim)
                .with_initializer(init.clone())
                .init(device),
            ffn_down: LinearConfig::new(config.hidden_dim, config.emb_dim)
                .with_initializer(init)
                .init(device),
            ffn_dropout: DropoutConfig::new(DROPOUT_RATE).init(),
            n_heads: config.n_heads,
            head_dim: config.head_dim,
        }
    }

    /// Forward pass: [batch, seq_len, d_model] -> [batch, seq_len, d_model]
    ///
    /// `cos` and `sin` are the precomputed rotary embedding tables of shape
    /// `[1, 1, seq_len, d_head]`, applied to Q and K to encode position.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: &Tensor<B, 4>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        self.forward_cached(x, Some(mask), cos, sin, &mut LayerCache::new())
    }

    /// Forward pass over `x`, attending to `cache` as well as to `x` itself, and appending
    /// the keys and values computed for `x` to it.
    ///
    /// With an empty cache and a mask this is an ordinary causal forward pass; with a
    /// non-empty cache and no mask it is a single decode step. See
    /// [`Model::forward_cached`] for why those are the only two shapes.
    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        mask: Option<&Tensor<B, 4>>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
        cache: &mut LayerCache<B>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _emb_dim] = x.dims();

        // Pre-norm + self-attention
        let normed = self.ln1.forward(x.clone());
        let qkv = self.attn_qkv.forward(normed);
        let qkv = qkv.reshape([batch, seq_len, 3, self.n_heads, self.head_dim]);

        let q = qkv.clone().slice([
            0..batch,
            0..seq_len,
            0..1,
            0..self.n_heads,
            0..self.head_dim,
        ]);
        let k = qkv.clone().slice([
            0..batch,
            0..seq_len,
            1..2,
            0..self.n_heads,
            0..self.head_dim,
        ]);
        let v = qkv.slice([
            0..batch,
            0..seq_len,
            2..3,
            0..self.n_heads,
            0..self.head_dim,
        ]);

        let q = q.reshape([batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.n_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.n_heads, self.head_dim]);

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // QK norm: normalize Q and K to prevent attention score explosion
        // RMS norm: x / sqrt(mean(x^2) + eps), applied along last dim (d_head)
        let q = Self::rms_norm(q);
        let k = Self::rms_norm(k);

        // Rotary position embeddings: rotate Q and K so their dot product depends on
        // relative position. Applied after QK norm so the rotation acts on unit-scale
        // vectors and only the phase (not magnitude) carries position information.
        let q = apply_rope(q, cos, sin);
        let k = apply_rope(k, cos, sin);

        // Prepend whatever the cache already holds, then put the extended keys and values
        // back for the next call. Cloning is a handle clone, not a copy.
        let (k, v) = match cache.kv.take() {
            Some((k_prev, v_prev)) => (
                Tensor::cat(vec![k_prev, k], 2),
                Tensor::cat(vec![v_prev, v], 2),
            ),
            None => (k, v),
        };
        cache.kv = Some((k.clone(), v.clone()));

        let scale = (self.head_dim as f32).sqrt();
        let k_t = k.swap_dims(2, 3);
        let attn = q.matmul(k_t) / scale;
        // A single query attends to the whole cache unconditionally: every cached position
        // precedes it, so there is nothing to mask out.
        let attn = match mask {
            Some(mask) => attn + mask.clone(),
            None => attn,
        };
        let attn = activation::softmax(attn, 3);
        let out = attn.matmul(v);

        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.n_heads * self.head_dim]);
        let out = self.attn_proj.forward(out);
        let out = self.attn_dropout.forward(out);

        let x = x + out;

        // Pre-norm + FFN
        let normed = self.ln2.forward(x.clone());
        let h = self.ffn_up.forward(normed);
        let h = activation::gelu(h);
        let h = self.ffn_down.forward(h);
        let h = self.ffn_dropout.forward(h);

        x + h
    }

    /// RMS normalization along the last dimension (no learnable parameters).
    ///
    /// Formula: `x / sqrt(mean(x^2) + eps)`
    ///
    /// Used for QK norm to prevent attention score explosion. By normalizing Q and K
    /// vectors before computing attention scores, the dot product becomes a cosine
    /// similarity bounded by the geometry rather than unbounded magnitudes.
    fn rms_norm<const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
        let eps = 1e-6;
        let x_sq = x.clone().powf_scalar(2.0);
        let mean_sq = x_sq.mean_dim(D - 1);
        let rms = (mean_sq + eps).sqrt();
        x / rms
    }
}

/// Wraps a [`Model`] with an optimizer for training.
///
/// Why is this separate from `Model`? The type system enforces it:
/// - `Model<B: Backend>` works with any backend (CPU, GPU, etc.)
/// - `Trainer<B: AutodiffBackend>` requires a backend that supports automatic differentiation
///
/// You can only call `.backward()` on tensors when the backend implements `AutodiffBackend`.
/// This is how Burn tracks operations during the forward pass and computes gradients via
/// the chain rule. The `Autodiff<B>` wrapper enables this:
///
/// ```ignore
/// type Inference = Wgpu<f32, i32>;           // Forward pass only
/// type Training = Autodiff<Wgpu<f32, i32>>;  // Forward + backward
/// ```
///
/// Keeping `Model` free of autodiff constraints means inference code doesn't carry
/// optimizer state or gradient tracking overhead.
pub struct Trainer<B: AutodiffBackend> {
    pub model: Model<B>,
    pub optimizer: OptimizerAdaptor<AdamW, Model<B>, B>,
    pub config: ModelConfig,
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(config: ModelConfig, device: &B::Device) -> Self {
        eprintln!(
            "Creating model: {} layers, emb_dim={}, heads={}",
            config.n_layers, config.emb_dim, config.n_heads
        );
        let model = Model::new(&config, device);

        eprintln!("  creating optimizer...");
        let optimizer = AdamWConfig::new()
            .with_weight_decay(0.1)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
            .init();

        Self {
            model,
            optimizer,
            config,
        }
    }

    pub fn from_model(model: Model<B>, config: ModelConfig, device: &B::Device) -> Self {
        let _ = device;
        eprintln!("  creating optimizer for fine-tuning...");
        let optimizer = AdamWConfig::new()
            .with_weight_decay(0.1)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
            .init();

        Self {
            model,
            optimizer,
            config,
        }
    }

    pub fn ctx_len(&self) -> usize {
        self.config.ctx_len
    }

    /// Training step: compute loss, backprop, update weights
    pub fn train_step(
        &mut self,
        input: Tensor<B, 2, Int>,
        target: Tensor<B, 2, Int>,
        lr: f64,
        device: &B::Device,
    ) -> f32 {
        let [batch, seq_len] = input.dims();

        let logits = self.model.forward(input, device);

        let logits = logits.reshape([batch * seq_len, self.config.vocab_size]);
        let target = target.reshape([batch * seq_len]);

        let loss = CrossEntropyLossConfig::new()
            .init(device)
            .forward(logits, target);

        let loss_val: f32 = loss.clone().into_scalar().elem();

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = self.optimizer.step(lr, self.model.clone(), grads);

        loss_val
    }

    /// Training step with masked loss: ignores positions where target == pad_id
    pub fn train_step_masked(
        &mut self,
        input: Tensor<B, 2, Int>,
        target: Tensor<B, 2, Int>,
        lr: f64,
        pad_id: usize,
        device: &B::Device,
    ) -> f32 {
        let [batch, seq_len] = input.dims();

        let logits = self.model.forward(input, device);

        let logits = logits.reshape([batch * seq_len, self.config.vocab_size]);
        let target = target.reshape([batch * seq_len]);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![pad_id]))
            .init(device)
            .forward(logits, target);

        let loss_val: f32 = loss.clone().into_scalar().elem();

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = self.optimizer.step(lr, self.model.clone(), grads);

        loss_val
    }

    /// Eval step: compute loss without gradients
    pub fn eval_step(
        &self,
        input: Tensor<B, 2, Int>,
        target: Tensor<B, 2, Int>,
        device: &B::Device,
    ) -> f32 {
        let [batch, seq_len] = input.dims();

        let inner = self.model.clone().valid();
        let logits = inner.forward(input.inner(), device);
        let logits = logits.reshape([batch * seq_len, self.config.vocab_size]);
        let target = target.inner().reshape([batch * seq_len]);

        let loss = CrossEntropyLossConfig::new()
            .init(device)
            .forward(logits, target);

        loss.into_scalar().elem()
    }

    /// Eval step with masked loss: ignores positions where target == pad_id
    pub fn eval_step_masked(
        &self,
        input: Tensor<B, 2, Int>,
        target: Tensor<B, 2, Int>,
        pad_id: usize,
        device: &B::Device,
    ) -> f32 {
        let [batch, seq_len] = input.dims();

        let inner = self.model.clone().valid();
        let logits = inner.forward(input.inner(), device);
        let logits = logits.reshape([batch * seq_len, self.config.vocab_size]);
        let target = target.inner().reshape([batch * seq_len]);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![pad_id]))
            .init(device)
            .forward(logits, target);

        loss.into_scalar().elem()
    }

    /// Save model weights and config
    pub fn save(&self, path: &Path) -> Result<()> {
        self.config.save(path)?;
        self.model
            .clone()
            .save_file(
                path,
                &burn::record::DefaultFileRecorder::<burn::record::FullPrecisionSettings>::new(),
            )
            .map_err(|e| crate::Error::Burn(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{NdArray, ndarray::NdArrayDevice};

    /// Logits for the final position of an uncached forward pass, which is what a cached
    /// decode step returns.
    fn last_row<B: Backend>(logits: Tensor<B, 3>) -> Vec<f32> {
        let [batch, seq_len, vocab] = logits.dims();

        logits
            .slice([0..batch, (seq_len - 1)..seq_len, 0..vocab])
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    }

    /// A deliberately awkward configuration: head_dim is not emb_dim / n_heads, so any
    /// place that still conflates the attention width with the embedding width fails.
    fn test_config() -> ModelConfig {
        ModelConfig {
            preset: None,
            n_layers: 2,
            emb_dim: 32,
            n_heads: 2,
            head_dim: 10,
            hidden_dim: 64,
            ctx_len: 16,
            vocab_size: 64,
        }
    }

    /// Stepping through a sequence with the cache must produce the same logits as running
    /// the whole prefix through the uncached path. This is the property that makes the
    /// cache an optimization rather than a change in behavior.
    #[test]
    fn cached_decode_matches_uncached_forward() {
        type B = NdArray<f32>;

        let device = NdArrayDevice::default();
        let config = test_config();
        let model = Model::<B>::new(&config, &device);
        let tokens: Vec<i32> = vec![3, 14, 15, 9, 26, 5, 35];

        let ids = |slice: &[i32]| {
            Tensor::<B, 2, Int>::from_data(
                TensorData::new(slice.to_vec(), [1, slice.len()]),
                &device,
            )
        };

        let expected = last_row(model.forward(ids(&tokens), &device));

        // Prefill everything but the last token, then decode that one from the cache.
        let (head, tail) = tokens.split_at(tokens.len() - 1);
        let mut cache = model.new_cache();
        model.forward_cached(ids(head), &mut cache, &device);
        let actual = model
            .forward_cached(ids(tail), &mut cache, &device)
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        assert_eq!(cache.len(), tokens.len());
        assert_eq!(expected.len(), actual.len());
        for (i, (want, got)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (want - got).abs() < 1e-4,
                "logit {i} diverged: uncached {want}, cached {got}",
            );
        }
    }

    /// Decoding one token at a time from an empty cache must also match, since that is how
    /// a caller feeds prompt tokens that arrive after the cache is already warm.
    #[test]
    fn token_at_a_time_matches_uncached_forward() {
        type B = NdArray<f32>;

        let device = NdArrayDevice::default();
        let config = test_config();
        let model = Model::<B>::new(&config, &device);
        let tokens: Vec<i32> = vec![7, 1, 42, 8, 19];

        let ids = |slice: &[i32]| {
            Tensor::<B, 2, Int>::from_data(
                TensorData::new(slice.to_vec(), [1, slice.len()]),
                &device,
            )
        };

        let expected = last_row(model.forward(ids(&tokens), &device));

        let mut cache = model.new_cache();
        let mut actual = Vec::new();
        for token in &tokens {
            actual = model
                .forward_cached(ids(&[*token]), &mut cache, &device)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
        }

        for (i, (want, got)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (want - got).abs() < 1e-4,
                "logit {i} diverged: uncached {want}, cached {got}",
            );
        }
    }

    /// Checkpoints trained before the preset work saved their config with the old
    /// field names and without head_dim/hidden_dim; those files must keep loading
    /// with the derivations that built them.
    #[test]
    fn loads_pre_preset_config_format() {
        let json = r#"{"layers":4,"d_model":256,"heads":4,"ctx_len":256,"vocab_size":50281}"#;
        let config: ModelConfig = serde_json::from_str(json).unwrap();

        assert_eq!(config.preset, None);
        assert_eq!(config.n_layers, 4);
        assert_eq!(config.emb_dim, 256);
        assert_eq!(config.n_heads, 4);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.hidden_dim, 1024);
        assert_eq!(config.ctx_len, 256);
        assert_eq!(config.vocab_size, 50281);
    }

    /// The preset name is stored in model.json as information and must survive a
    /// save/load round trip with its proper spelling.
    #[test]
    fn preset_name_round_trips_through_json() {
        let json = serde_json::to_string(&ModelConfig::somen()).unwrap();
        let config: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.preset.as_deref(), Some("Sōmen"));
    }
}
