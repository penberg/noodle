# Theory of Operation

## Summary

Noodle is a minimal and clean small language model, following the GPT-2 decoder-only transformer architecture. The implementation closely follows the original GPT-2 paper with pre-LayerNorm, learned positional embeddings, and GELU activations.

## Command Line Interface

Noodle provides two commands:

- **train** — Takes a text corpus, produces a model file
- **chat** — Interactive generation loop

Model files use Burn's MessagePack format (`.mpk`) for weights, with a companion JSON file (`.json`) storing architecture configuration (layer count, dimensions, vocabulary size).

Design principles:
- Model file is always a positional argument, not a flag
- Prompts can come from positional arg, `--prompt` flag, or stdin
- All hyperparameters have sensible defaults
- Progress output goes to stderr, generated text to stdout

## Architecture

Noodle has a single canonical model configuration, sized for small datasets (~300K-1M tokens):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Layers | 4 | Transformer blocks |
| d_model | 256 | Hidden dimension |
| Heads | 4 | Attention heads (64 dims/head) |
| Context | 256 | Maximum sequence length |
| Vocab | 50,281 | p50k_base tokenizer |
| Dropout | 0.1 | Applied after attention and FFN |

This yields ~29M parameters — small enough to train on consumer hardware in minutes, appropriately sized for small datasets to avoid overfitting.

### Parameter Count Breakdown

| Component | Parameters | Formula |
|-----------|------------|---------|
| Token embeddings | 12,872,000 | vocab_size × d_model = 50,281 × 256 |
| Position embeddings | 65,536 | ctx_len × d_model = 256 × 256 |
| Per-block (×4 blocks): | | |
| — LN1 (weight + bias) | 512 | 2 × d_model |
| — QKV projection | 197,376 | d_model × 3×d_model + 3×d_model |
| — Attention output | 65,792 | d_model × d_model + d_model |
| — LN2 (weight + bias) | 512 | 2 × d_model |
| — FFN up | 263,168 | d_model × 4×d_model + 4×d_model |
| — FFN down | 262,400 | 4×d_model × d_model + d_model |
| Final layer norm | 512 | 2 × d_model |
| Output projection | 12,872,281 | d_model × vocab_size + vocab_size |

### Regularization

To prevent overfitting on small datasets, Noodle uses:
- **Dropout (0.1)** — Applied after attention projection and FFN output
- **Weight decay (0.1)** — AdamW optimizer penalizes large weights
- **Gradient clipping (1.0)** — Prevents exploding gradients
- **Train/val split (90/10)** — Early stopping based on validation loss
- **Best model checkpointing** — Saves the model with lowest validation loss

## Components

### Tokenizer

Noodle uses [tiktoken-rs](https://github.com/zurawiki/tiktoken-rs) with the `p50k_base` vocabulary (~50K tokens). This is the same tokenizer used by GPT-3 and Codex. Tokenization is a separate problem from model architecture — no need to reinvent this wheel.

### Embeddings

Two embedding tables convert discrete inputs to continuous vectors:

1. **Token embeddings** (`token_emb`): Maps each vocabulary token to a d_model-dimensional vector. Initialized with Normal(μ=0, σ=0.02).

2. **Position embeddings** (`pos_emb`): Learned embeddings for each position 0..ctx_len. Unlike the original Transformer's sinusoidal encodings, GPT uses learned positional embeddings. Also initialized with Normal(0, 0.02).

The forward pass adds these: `x = token_emb[tokens] + pos_emb[0:seq_len]`

### Transformer Block

Each of the 4 transformer blocks follows the **pre-norm** architecture (GPT-2 style):

```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

This differs from the original Transformer's post-norm (`x = LayerNorm(x + Sublayer(x))`). Pre-norm is more stable for training deep networks.

#### Self-Attention

Multi-head scaled dot-product attention with causal masking:

1. **QKV Projection**: A single linear layer projects the input to Q, K, V concatenated:
   ```
   qkv = Linear(d_model, 3 × d_model)(LayerNorm(x))
   ```

2. **Split and reshape** to [batch, heads, seq_len, d_head]:
   ```
   Q, K, V = split(qkv, 3)
   Q = Q.reshape(batch, seq_len, heads, d_head).transpose(1, 2)
   ```

3. **Scaled dot-product attention**:
   ```
   scores = (Q @ K.T) / sqrt(d_head)
   scores = scores + causal_mask  # -1e9 for future positions
   attn = softmax(scores, dim=-1)
   out = attn @ V
   ```

4. **Output projection**: Linear back to d_model, then dropout:
   ```
   out = Dropout(Linear(d_model, d_model)(out.reshape(...)))
   ```

The causal mask ensures position i can only attend to positions ≤ i, enforcing the autoregressive property.

#### Feed-Forward Network (MLP)

A two-layer MLP with GELU activation and 4× expansion:

```
h = GELU(Linear(d_model, 4 × d_model)(LayerNorm(x)))
out = Dropout(Linear(4 × d_model, d_model)(h))
```

GELU (Gaussian Error Linear Unit) provides smoother gradients than ReLU:
```
GELU(x) = x × Φ(x)  where Φ is the standard Gaussian CDF
```

### Layer Normalization

Applied **before** each sublayer (pre-norm). Normalizes across the feature dimension:

```
y = (x - mean(x)) / sqrt(var(x) + ε) × γ + β
```

where γ (weight) and β (bias) are learned per-dimension.

A final layer norm (`ln_f`) is applied after all transformer blocks, before the output projection.

### Output Projection

A linear layer maps from d_model to vocab_size, producing logits for each token:

```
logits = Linear(d_model, vocab_size)(LayerNorm(x))
```

No weight tying is used between input embeddings and output projection.

## Training

### Loss Function

Cross-entropy loss between predicted logits and target token IDs. For next-token prediction:

- **Input**: tokens[0:n-1]
- **Target**: tokens[1:n]
- **Loss**: `-log(softmax(logits)[target])`

The loss is computed over all positions in the sequence, then averaged.

### Optimizer

AdamW (Adam with decoupled weight decay):
- Learning rate: 1e-4 (constant)
- Weight decay: 0.1 (applied to weights, not biases)
- Gradient clipping: max norm 1.0

### Data Batching

Following the nanochat approach, training uses non-overlapping consecutive chunks:

1. Split corpus into train (90%) and validation (10%) sets
2. Stream through tokens in order, taking `batch_size × ctx_len` tokens per batch
3. Input = tokens[0:n], Target = tokens[1:n+1] (shifted by one)
4. No shuffling — preserves document structure

### Early Stopping

Training stops when validation loss doesn't improve by `MIN_IMPROVEMENT` (0.01) for `PATIENCE` (5) epochs. The best model (lowest validation loss) is saved.

## Inference

### Autoregressive Generation

Tokens are generated one at a time, each conditioned on all previous tokens:

```
for _ in range(max_tokens):
    logits = model.forward(tokens)  # [batch, seq, vocab]
    next_logit = logits[:, -1, :]   # last position
    next_token = sample(next_logit)
    tokens.append(next_token)
```

Context is truncated to `ctx_len` if the sequence grows too long.

### Sampling Strategies

Noodle implements several sampling strategies that can be combined:

1. **Temperature** (default: 0.7): Scales logits before softmax. Lower = more deterministic, higher = more random.
   ```
   logits = logits / temperature
   ```

2. **Top-k** (default: 40): Keep only the k highest-probability tokens, zero out the rest.
   ```
   top_k_logits = logits[argsort(logits)[-k:]]
   ```

3. **Top-p / Nucleus** (default: 0.95): Keep the smallest set of tokens whose cumulative probability ≥ p.
   ```
   sorted_probs = sort(softmax(logits))
   cumsum = cumulative_sum(sorted_probs)
   nucleus = probs where cumsum < p
   ```

4. **Repetition penalty** (default: 1.1): Reduce probability of tokens that already appeared.
   ```
   for token in context:
       if logits[token] > 0:
           logits[token] /= penalty
       else:
           logits[token] *= penalty
   ```

### KV Cache

*Not yet implemented.* Currently, each generation step recomputes attention for all previous tokens. A KV cache would store the key and value tensors from previous positions, reducing generation from O(n²) to O(n) per token.

## References

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017). "[Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)". In _NeurIPS 2017_.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever (2018). "[Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)".

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever (2019). "[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)".
