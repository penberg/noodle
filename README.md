<p align="center">
  <img src=".github/assets/noodle.png" alt="Noodle" width="400px">
</p>

<h1 align="center">Noodle 🍜</h1>

<p align="center">
 A small language model. 
</p>

## Introduction

Noodle is a minimal language model implementation written in Rust. It's designed to be simple, educational, and easy to understand — a from-scratch implementation of a transformer-based language model that you can train on your own data.

**Features:**

- Transformer architecture with configurable layers and embedding dimensions
- Training on custom text corpora
- Interactive chat interface for text generation
- GPU acceleration support

## Getting Started

### Download training corpus

```console
> mkdir -p corpus
> curl -L https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt -o corpus/tinystories-train.txt
```

### Training the model

First, do a quick training pass with a small testing dataset to verify everything is working:

```console
> head -10000 corpus/tinystories-train.txt > corpus/tinystories-test.txt
> cargo run --release -- train corpus/tinystories-test.txt models/noodle --max-epochs 20
```

Then, train the model with the full dataset:

```console
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

```console
> uv run modal volume put noodle-data corpus/tinystories-test.txt /corpus/
> uv run modal run jobs/modal/train.py --corpus-file /data/corpus/tinystories-test.txt --max-epochs 1
```

**Run full training on Modal GPU:**

```console
> uv run modal volume put noodle-data corpus/tinystories-train.txt /corpus/
> uv run modal run jobs/modal/train.py --corpus-file /data/corpus/tinystories-train.txt --max-epochs 20
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
  creating position embeddings...
  creating 4 transformer blocks...
  creating final layer norm...
  creating output projection...
  model ready

 ~(°◡°)~  Noodle

I am ready to chat! Type your message and press Enter.

> Once upon a time
, there was a little girl named Lily. She loved to design things with her crayons. One day, she wanted to design something new and pretty. So, she added some colors and shapes in it.Lily wanted to create something special for her mom. She put pretty colors on the paper and made a beautiful picture of a beautiful picture on the paper with many colors on it. When she finished drawing, they were very happy with their creative work - not just like Lily's painting!
```

Note that the model is only pre-trained and, therefore, only performs text generation, and does not answer questions. For that, we need finetuning support.

## License

This project is licensed under the [MIT license].

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in Weave by you, shall be licensed as MIT, without any additional
terms or conditions.

[MIT license]: LICENSE.md
