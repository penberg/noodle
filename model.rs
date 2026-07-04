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

/// Configuration for the decoder-only transformer model architecture.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub layers: usize,
    pub d_model: usize,
    pub heads: usize,
    pub ctx_len: usize,
    pub vocab_size: usize,
}

impl ModelConfig {
    pub fn new(
        layers: usize,
        d_model: usize,
        heads: usize,
        ctx_len: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            layers,
            d_model,
            heads,
            ctx_len,
            vocab_size,
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
    d_head: usize,
}

impl<B: Backend> Model<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        eprintln!("  initializing...");

        // Rotary position embeddings rotate the head dimension in pairs, so d_head must
        // be even (and d_model must divide evenly into heads). Fail early with a clear
        // message rather than panicking on a shape mismatch deep in the forward pass.
        assert!(
            config.d_model.is_multiple_of(config.heads),
            "d_model ({}) must be divisible by heads ({})",
            config.d_model,
            config.heads,
        );
        let d_head = config.d_model / config.heads;
        assert!(
            d_head.is_multiple_of(2),
            "d_head (d_model / heads = {d_head}) must be even for rotary position embeddings",
        );

        eprintln!(
            "  creating token embeddings ({} x {})...",
            config.vocab_size, config.d_model
        );
        let emb_init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        let token_emb = EmbeddingConfig::new(config.vocab_size, config.d_model)
            .with_initializer(emb_init)
            .init(device);

        eprintln!("  using rotary position embeddings (RoPE)");

        eprintln!("  creating {} transformer blocks...", config.layers);
        let mut blocks = Vec::with_capacity(config.layers);
        for _ in 0..config.layers {
            let block = TransformerBlock::new(config.d_model, config.heads, device);
            blocks.push(block);
        }

        eprintln!("  creating final layer norm...");
        let ln_f = LayerNormConfig::new(config.d_model).init(device);

        eprintln!("  creating output projection...");
        let init = Initializer::XavierUniform { gain: 1.0 };
        let output = LinearConfig::new(config.d_model, config.vocab_size)
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
            d_head,
        }
    }

    pub fn load(path: &Path, device: &B::Device) -> Result<Self> {
        let config = ModelConfig::load(path)?;

        eprintln!(
            "Loading model: {} layers, d_model={}",
            config.layers, config.d_model
        );

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

        // Rotary position embeddings: precompute the cos/sin tables once and share
        // them across all blocks. Position information is injected by rotating Q and K
        // inside attention rather than by adding an absolute position embedding here.
        let (cos, sin) = rope_tables::<B>(seq_len, self.d_head, device);

        // Causal mask [1, 1, seq_len, seq_len]: 0 for attend, -1e9 for mask
        // triu_mask returns FALSE for upper triangle (future), TRUE for lower triangle (past/current)
        // (it's a "mask for upper triangle operation", not "upper triangle is true")
        let attn_mask: Tensor<B, 2, Bool> = Tensor::triu_mask([seq_len, seq_len], 1, device);
        let zeros: Tensor<B, 2> = Tensor::zeros([seq_len, seq_len], device);
        let large_neg: Tensor<B, 2> = Tensor::full([seq_len, seq_len], -1e9f32, device);
        // mask_where: self.mask_where(mask, value) → value where TRUE, self where FALSE
        // TRUE (past/current, j ≤ i) → zeros (can attend)
        // FALSE (future, j > i) → large_neg (blocked)
        let mask = large_neg.mask_where(attn_mask, zeros);
        let mask = mask.reshape([1, 1, seq_len, seq_len]);

        let mut x = self.token_emb.forward(token_ids);

        for block in &self.blocks {
            x = block.forward(x, &mask, &cos, &sin);
        }

        let x = self.ln_f.forward(x);
        self.output.forward(x)
    }
}

/// Base frequency for rotary position embeddings (RoPE), following the original paper.
const ROPE_BASE: f32 = 10_000.0;

/// Precompute the RoPE cosine and sine tables for a given sequence length.
///
/// Rotary position embeddings encode absolute position by rotating pairs of dimensions
/// in the Q and K vectors by an angle proportional to the position. Dimension pair `i`
/// rotates at frequency `ROPE_BASE^(-2i/d_head)`, so low dimensions rotate slowly (coarse
/// position) and high dimensions rotate quickly (fine position). Because the rotation is a
/// linear map, the attention dot product `q·k` ends up depending only on the *relative*
/// distance between the two positions — the property that makes RoPE generalize across
/// sequence lengths without any learned position parameters.
///
/// Returns `(cos, sin)`, each of shape `[1, 1, seq_len, d_head]` so they broadcast over the
/// `[batch, heads, seq_len, d_head]` Q/K tensors. This uses the "rotate-half" layout
/// (as in LLaMA/GPT-NeoX): the frequency vector is duplicated so the first and second
/// halves of `d_head` share angles, pairing dimension `i` with dimension `i + d_head/2`.
fn rope_tables<B: Backend>(
    seq_len: usize,
    d_head: usize,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let half = d_head / 2;

    // inv_freq[i] = ROPE_BASE^(-2i/d_head) for i in 0..half
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| ROPE_BASE.powf(-(2.0 * i as f32) / d_head as f32))
        .collect();
    let inv_freq = Tensor::<B, 1>::from_data(TensorData::new(inv_freq, [half]), device);

    let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
    let positions = Tensor::<B, 1>::from_data(TensorData::new(positions, [seq_len]), device);

    // Outer product: freqs[p, i] = position p * inv_freq[i] -> [seq_len, half]
    let freqs = positions.reshape([seq_len, 1]) * inv_freq.reshape([1, half]);

    // Duplicate along the feature dim so angles line up with the rotate-half layout.
    let emb = Tensor::cat(vec![freqs.clone(), freqs], 1); // [seq_len, d_head]

    let cos = emb.clone().cos().reshape([1, 1, seq_len, d_head]);
    let sin = emb.sin().reshape([1, 1, seq_len, d_head]);
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
    heads: usize,
    d_head: usize,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(d_model: usize, heads: usize, device: &B::Device) -> Self {
        let d_head = d_model / heads;
        let init = Initializer::XavierUniform { gain: 1.0 };
        Self {
            ln1: LayerNormConfig::new(d_model).init(device),
            attn_qkv: LinearConfig::new(d_model, 3 * d_model)
                .with_initializer(init.clone())
                .init(device),
            attn_proj: LinearConfig::new(d_model, d_model)
                .with_initializer(init.clone())
                .init(device),
            attn_dropout: DropoutConfig::new(DROPOUT_RATE).init(),
            ln2: LayerNormConfig::new(d_model).init(device),
            ffn_up: LinearConfig::new(d_model, 4 * d_model)
                .with_initializer(init.clone())
                .init(device),
            ffn_down: LinearConfig::new(4 * d_model, d_model)
                .with_initializer(init)
                .init(device),
            ffn_dropout: DropoutConfig::new(DROPOUT_RATE).init(),
            heads,
            d_head,
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
        let [batch, seq_len, d_model] = x.dims();

        // Pre-norm + self-attention
        let normed = self.ln1.forward(x.clone());
        let qkv = self.attn_qkv.forward(normed);
        let qkv = qkv.reshape([batch, seq_len, 3, self.heads, self.d_head]);

        let q = qkv
            .clone()
            .slice([0..batch, 0..seq_len, 0..1, 0..self.heads, 0..self.d_head]);
        let k = qkv
            .clone()
            .slice([0..batch, 0..seq_len, 1..2, 0..self.heads, 0..self.d_head]);
        let v = qkv.slice([0..batch, 0..seq_len, 2..3, 0..self.heads, 0..self.d_head]);

        let q = q.reshape([batch, seq_len, self.heads, self.d_head]);
        let k = k.reshape([batch, seq_len, self.heads, self.d_head]);
        let v = v.reshape([batch, seq_len, self.heads, self.d_head]);

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

        let scale = (self.d_head as f32).sqrt();
        let k_t = k.swap_dims(2, 3);
        let attn = q.matmul(k_t) / scale;
        let attn = attn + mask.clone();
        let attn = activation::softmax(attn, 3);
        let out = attn.matmul(v);

        let out = out.swap_dims(1, 2).reshape([batch, seq_len, d_model]);
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
            "Creating model: {} layers, d_model={}, heads={}",
            config.layers, config.d_model, config.heads
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
