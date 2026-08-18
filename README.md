<p align="center">
  <img src=".github/assets/noodle.png" alt="Noodle" width="400px">
</p>

<h1 align="center">Noodle 🍜</h1>

<p align="center">
An on-device, GPU-accelerated language model in Rust. 
</p>

## Introduction

Noodle is a language model implemented from scratch in Rust. It's a decoder-only transformer with a complete pipeline — training, fine-tuning on instruction data, evaluation, and an interactive inference runtime — plus cloud-GPU training jobs on Modal. No PyTorch and no Python model code: the entire stack is Rust, with GPU acceleration.

## Getting Started

### Training the model

Noodle streams the training corpus directly from HuggingFace — no download step needed. First, do a quick training pass with the small validation split to verify everything is working:

```console
> cargo run --release -- train hf://roneneldan/TinyStories/TinyStoriesV2-GPT4-valid.txt models/noodle --max-epochs 1
```

Then, train the model with the full dataset:

```console
> cargo run --release -- train hf://roneneldan/TinyStories/TinyStoriesV2-GPT4-train.txt models/noodle --max-epochs 20
```

The corpus argument accepts an `hf://owner/repo/file` spec (streamed from a HuggingFace dataset repository), any `http(s)://` URL, or a local file path.

### Training from local files

You can also download the corpus and train from a local file:

```console
> mkdir -p corpus
> curl -L https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt -o corpus/tinystories-train.txt
> head -10000 corpus/tinystories-train.txt > corpus/tinystories-test.txt
> cargo run --release -- train corpus/tinystories-train.txt models/noodle --max-epochs 20
```

### Training on cloud GPUs (Modal)

For faster training, you can use [Modal](https://modal.com/) to train on cloud GPUs.

**Prerequisites:**

```console
> uv pip install modal
> uv run modal setup
```

**Create Modal volume:**

```console
> uv run modal volume create noodle-data
```

**Test with small dataset first:**

The corpus is streamed from HuggingFace by default, so no upload step is needed:

```console
> uv run modal run jobs/modal/train.py --corpus hf://roneneldan/TinyStories/TinyStoriesV2-GPT4-valid.txt --max-epochs 1
```

**Run full training on Modal GPU:**

```console
> uv run modal run jobs/modal/train.py --max-epochs 20
```

**Train on a local corpus file:**

You can also upload a corpus file to the volume and train on that:

```console
> uv run modal volume put noodle-data corpus/tinystories-train.txt /corpus/
> uv run modal run jobs/modal/train.py --corpus /data/corpus/tinystories-train.txt --max-epochs 20
```

**Download trained model to local machine:**

```console
> mkdir -p models/noodle
> uv run modal volume get noodle-data /models/noodle/model.json ./models/noodle/
> uv run modal volume get noodle-data /models/noodle/model.mpk ./models/noodle/
```

You can inspect the volume contents with:

```console
> uv run modal run jobs/modal/train.py::list_volume
```

### Running inference

When you have a trained model, you can use it for inference:

```console
> cargo run --release chat models/noodle/model.mpk
Using GPU device: DefaultDevice
Loading model: 4 layers, d_model=256
  initializing...
  creating token embeddings (50281 x 256)...
  using rotary position embeddings (RoPE)
  creating 4 transformer blocks...
  creating final layer norm...
  creating output projection...
  model ready

 ~(°◡°)~  Noodle

I am ready to chat! Type your message and press Enter.

> Once upon a time
, there was a little girl named Lily. She loved to design things with her crayons. One day, she wanted to design something new and pretty. So, she added some colors and shapes in it.Lily wanted to create something special for her mom. She put pretty colors on the paper and made a beautiful picture of a beautiful picture on the paper with many colors on it. When she finished drawing, they were very happy with their creative work - not just like Lily's painting!
```

Note that the pre-trained model only performs text generation and does not follow instructions. To make the model follow instructions, you can fine-tune it on instruction data (see below).

### Fine-tuning the model

You can fine-tune a pre-trained model on instruction data to teach it to follow instructions:

```console
> cargo run --release -- finetune models/noodle/model.mpk corpus/instructions.txt models/noodle-finetuned --max-epochs 5
```

The fine-tuning command takes the following arguments:

- `model` — Path to the pre-trained model `.mpk` file
- `input` — Instruction text file (training data): a local path, `hf://owner/repo/file` spec, or `http(s)://` URL
- `output` — Output directory for the fine-tuned model

Optional flags:

- `--backend` — Backend to use for training: `gpu` (default), `cuda`, or `cpu`.
  Defaults to `$NOODLE_BACKEND` when set
- `--max-epochs` — Maximum number of fine-tuning epochs (default: `5`)

The fine-tuned model is saved to the output directory and can be used with the `chat` command:

```console
> cargo run --release chat models/noodle-finetuned/model.mpk
```

## Model Architecture

Noodle is a pre-norm, decoder-only transformer: rotary position embeddings, parameter-free RMS norm on the queries and keys, and a GELU feed-forward network. The diagram below shows the full stack on the left and a single transformer block expanded on the right — names in monospace are the fields in [`model.rs`](model.rs), and the sizes are the default configuration from [`train.rs`](train.rs).

<p align="center">
  <img src="docs/model.svg" alt="Noodle model architecture: input tokens flow through a token embedding, four pre-norm transformer blocks, a final layer norm and an output projection to logits. Each block applies a fused QKV linear, parameter-free RMS norm on Q and K, rotary position embeddings, masked softmax attention and a GELU feed-forward network, both wrapped in residual connections." width="900px">
</p>

## Benchmarks

Noodle ships [Criterion](https://github.com/bheisler/criterion.rs) benchmarks for the two
hot paths: inference and training. Run them with:

```console
> cargo bench                      # both suites
> cargo bench --bench inference    # forward pass and single-token generation
> cargo bench --bench training     # train step and eval step
```

The `inference` suite measures the model's forward pass and `generate_next_token` (forward
pass plus repetition penalty and top-k/top-p sampling). The `training` suite measures a
full optimizer step — forward, loss, backward, AdamW update — alongside the forward-only
validation step, so the cost of the backward pass and the optimizer update is visible by
comparison.

Every benchmark uses the canonical Noodle hyperparameters (4 layers, `d_model=256`,
4 heads, p50k_base vocabulary) on randomly initialized weights, and runs at a full
256-token context with the batch size its caller uses: 1 for inference, 8 for training.

Like the `noodle` commands themselves, benchmarks run on the wgpu backend by default.
Benchmarks have no command line of their own, so they take the backend from
`NOODLE_BACKEND` — the same variable the `noodle` commands read when `--backend` is
omitted:

```console
> NOODLE_BACKEND=cuda cargo bench --bench training   # CUDA
> NOODLE_BACKEND=cpu cargo bench --bench inference   # CPU (ndarray)
```

The backend name is part of every benchmark id, so results from different backends don't
overwrite each other's baselines.

You can filter to a single case by name, which is useful while iterating on one code path:

```console
> cargo bench --bench inference -- forward
```

Criterion compares each run against the previous one and reports the change, so the usual
workflow is to benchmark on `main`, apply a change, and benchmark again. Full results,
including plots, land in `target/criterion/`.

## License

This project is licensed under the [MIT license].

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in Noodle by you, shall be licensed as MIT, without any additional
terms or conditions.

[MIT license]: LICENSE.md
