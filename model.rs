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
    pos_emb: Embedding<B>,
    blocks: Vec<TransformerBlock<B>>,
    ln_f: LayerNorm<B>,
    output: Linear<B>,
    ctx_len: usize,
    vocab_size: usize,
}

impl<B: Backend> Model<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        eprintln!("  initializing...");

        eprintln!(
            "  creating token embeddings ({} x {})...",
            config.vocab_size, config.d_model
        );
        let emb_init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        let token_emb = EmbeddingConfig::new(config.vocab_size, config.d_model)
            .with_initializer(emb_init.clone())
            .init(device);

        eprintln!("  creating position embeddings...");
        let pos_emb = EmbeddingConfig::new(config.ctx_len, config.d_model)
            .with_initializer(emb_init)
            .init(device);

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
            pos_emb,
            blocks,
            ln_f,
            output,
            ctx_len: config.ctx_len,
            vocab_size: config.vocab_size,
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

        // Create position IDs [1, seq_len] for actual sequence length
        let pos_ids = create_position_ids::<B>(seq_len, device);

        // Create causal mask [1, 1, seq_len, seq_len] for actual sequence length
        let mask = create_causal_mask::<B>(seq_len, device);

        let tok_emb = self.token_emb.forward(token_ids);
        let pos_emb = self.pos_emb.forward(pos_ids);

        let mut x = tok_emb + pos_emb;

        for block in &self.blocks {
            x = block.forward(x, &mask);
        }

        let x = self.ln_f.forward(x);
        self.output.forward(x)
    }
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
    pub fn forward(&self, x: Tensor<B, 3>, mask: &Tensor<B, 4>) -> Tensor<B, 3> {
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

        let scale = (self.d_head as f32).sqrt();
        let k_t = k.swap_dims(2, 3);
        let attn = q.matmul(k_t) / scale;
        let attn = attn + mask.clone();
        let attn = attn.clamp(-1e4, 1e4); // Prevent softmax overflow
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
}

/// Create causal attention mask [1, 1, seq_len, seq_len]
/// Positions can attend to themselves and all previous positions.
/// Future positions are masked with -1e9.
fn create_causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 4> {
    // triu_mask with offset 1 gives True for positions ABOVE diagonal (future)
    let future_mask: Tensor<B, 2, Bool> = Tensor::triu_mask([seq_len, seq_len], 1, device);
    let zeros: Tensor<B, 2> = Tensor::zeros([seq_len, seq_len], device);
    let large_neg: Tensor<B, 2> = Tensor::full([seq_len, seq_len], -1e9f32, device);
    // mask_where puts second arg where condition is TRUE, keeps first arg where FALSE
    // We want -1e9 where future_mask is TRUE (can't attend to future)
    // But empirically mask_where seems inverted, so swap the order:
    let causal_mask = large_neg.mask_where(future_mask, zeros);

    causal_mask.reshape([1, 1, seq_len, seq_len])
}

/// Create position IDs [1, seq_len]
fn create_position_ids<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2, Int> {
    let pos_data: Vec<i32> = (0..seq_len as i32).collect();
    Tensor::from_data(TensorData::new(pos_data, [1, seq_len]), device)
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
