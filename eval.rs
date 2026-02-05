use std::{path::Path, time::Instant};

use burn::{
    backend::{Cuda, NdArray, Wgpu, cuda::CudaDevice, ndarray::NdArrayDevice, wgpu::WgpuDevice},
    nn::loss::CrossEntropyLossConfig,
    prelude::Backend,
    tensor::{ElementConversion, Int, Tensor, TensorData},
};

use crate::{
    Result,
    model::{Model, ModelConfig},
    tokenizer::Tokenize,
};

const BATCH_SIZE: usize = 8;
const LOG_INTERVAL: usize = 100;

pub fn eval(model_path: &Path, corpus_path: &Path, backend: crate::Backend) -> Result<()> {
    #[cfg(debug_assertions)]
    eprintln!("Warning: running in debug mode, use --release for faster evaluation");

    let config = ModelConfig::load(model_path)?;
    let ctx_len = config.ctx_len;

    eprintln!("Evaluating model on {}", corpus_path.display());

    let tokens = corpus_path.tokenize()?;
    let num_tokens = tokens.len();

    let tokens_per_batch = BATCH_SIZE * ctx_len;
    let num_batches = (tokens.len() - 1) / tokens_per_batch;

    if num_batches == 0 {
        return Err(crate::Error::Burn(format!(
            "Not enough tokens ({}) for batch size {} x context {}. Need at least {} tokens.",
            tokens.len(),
            BATCH_SIZE,
            ctx_len,
            tokens_per_batch + 1
        )));
    }

    eprintln!("  Tokens: {:}", format_number(num_tokens));
    eprintln!("  Batches: {:}", format_number(num_batches));
    eprintln!();

    match backend {
        crate::Backend::Wgpu => {
            let device = WgpuDevice::default();
            eprintln!("Using wgpu device: {:?}", device);
            eval_loop::<Wgpu<f32, i32>>(model_path, &tokens, ctx_len, num_batches, device)
        }
        crate::Backend::Cuda => {
            let device = CudaDevice::default();
            eprintln!("Using CUDA device: {:?}", device);
            eval_loop::<Cuda<f32, i32>>(model_path, &tokens, ctx_len, num_batches, device)
        }
        crate::Backend::Cpu => {
            let device = NdArrayDevice::default();
            eprintln!("Using CPU device: {:?}", device);
            eval_loop::<NdArray<f32>>(model_path, &tokens, ctx_len, num_batches, device)
        }
    }
}

fn eval_loop<B: Backend>(
    model_path: &Path,
    tokens: &[u32],
    ctx_len: usize,
    num_batches: usize,
    device: B::Device,
) -> Result<()> {
    let model: Model<B> = Model::load(model_path, &device)?;
    let vocab_size = model.vocab_size();

    let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let tokens_per_batch = BATCH_SIZE * ctx_len;

    eprintln!();
    eprintln!("Evaluating...");
    let start = Instant::now();
    let mut total_loss = 0.0f64;

    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * tokens_per_batch;

        let mut input_data = Vec::with_capacity(tokens_per_batch);
        let mut target_data = Vec::with_capacity(tokens_per_batch);

        for i in 0..tokens_per_batch {
            input_data.push(token_ids[batch_start + i]);
            target_data.push(token_ids[batch_start + i + 1]);
        }

        let input: Tensor<B, 2, Int> =
            Tensor::from_data(TensorData::new(input_data, [BATCH_SIZE, ctx_len]), &device);
        let target: Tensor<B, 2, Int> =
            Tensor::from_data(TensorData::new(target_data, [BATCH_SIZE, ctx_len]), &device);

        let logits = model.forward(input, &device);
        let logits = logits.reshape([BATCH_SIZE * ctx_len, vocab_size]);
        let target = target.reshape([BATCH_SIZE * ctx_len]);

        let loss = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(logits, target);

        let loss_val: f32 = loss.into_scalar().elem();
        total_loss += loss_val as f64;

        if (batch_idx + 1) % LOG_INTERVAL == 0 || batch_idx == 0 {
            let elapsed = start.elapsed().as_secs_f32();
            let batches_per_sec = (batch_idx + 1) as f32 / elapsed;
            eprintln!(
                "  batch {}/{}: loss = {:.4}, {:.1} batches/s",
                batch_idx + 1,
                num_batches,
                loss_val,
                batches_per_sec
            );
        }
    }

    let elapsed = start.elapsed();
    let avg_loss = total_loss / num_batches as f64;
    let perplexity = (avg_loss).exp();
    let tokens_evaluated = num_batches * tokens_per_batch;
    let throughput = tokens_evaluated as f64 / elapsed.as_secs_f64();

    eprintln!();
    eprintln!("Results:");
    eprintln!("  Loss: {:.4}", avg_loss);
    eprintln!("  Perplexity: {:.2}", perplexity);
    eprintln!("  Tokens evaluated: {}", format_number(tokens_evaluated));
    eprintln!("  Throughput: {} tokens/s", format_number(throughput as usize));

    Ok(())
}

fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}
