use std::{path::Path, time::Instant};

use burn::{
    backend::{
        Autodiff, Cuda, NdArray, Wgpu, cuda::CudaDevice, ndarray::NdArrayDevice, wgpu::WgpuDevice,
    },
    module::AutodiffModule,
    tensor::{Int, Tensor, TensorData, backend::AutodiffBackend},
};

use crate::{
    Result,
    model::{Model, ModelConfig, Trainer},
    tokenizer::Tokenize,
};

// Fine-tuning hyperparameters (lower LR and batch size than pretraining)
const BATCH_SIZE: usize = 4;
const LOG_INTERVAL: usize = 10;
const BASE_LEARNING_RATE: f64 = 2e-5;
const BASE_BATCH_SIZE: usize = 4;

// Early stopping: tighter than pretraining since SFT converges faster
const PATIENCE: usize = 3;
const MIN_IMPROVEMENT: f32 = 0.005;

// Train/validation split ratio (90% train, 10% validation)
const VAL_SPLIT: f32 = 0.1;

pub fn finetune(
    model_path: &Path,
    input: &Path,
    output: &Path,
    backend: crate::Backend,
    max_epochs: usize,
) -> Result<()> {
    #[cfg(debug_assertions)]
    eprintln!("Warning: running in debug mode, use --release for faster fine-tuning");

    // Create output directory
    std::fs::create_dir_all(output)?;

    // Copy model config to output directory so fine-tuned model is self-contained
    let config = ModelConfig::load(model_path)?;
    let output_model_path = output.join("model.mpk");
    config.save(&output_model_path)?;
    eprintln!("Saved config to {}", output.join("model.json").display());

    let tokens = input.tokenize()?;
    eprintln!("Loaded {} tokens", tokens.len());

    match backend {
        crate::Backend::Wgpu => {
            let device = WgpuDevice::default();
            eprintln!("Using wgpu device: {:?}", device);
            finetune_loop::<Autodiff<Wgpu<f32, i32>>>(
                model_path, config, &tokens, output, device, max_epochs,
            )
        }
        crate::Backend::Cuda => {
            let device = CudaDevice::default();
            eprintln!("Using CUDA device: {:?}", device);
            finetune_loop::<Autodiff<Cuda<f32, i32>>>(
                model_path, config, &tokens, output, device, max_epochs,
            )
        }
        crate::Backend::Cpu => {
            let device = NdArrayDevice::default();
            eprintln!("Using CPU device: {:?}", device);
            finetune_loop::<Autodiff<NdArray<f32>>>(
                model_path, config, &tokens, output, device, max_epochs,
            )
        }
    }
}

fn finetune_loop<B: AutodiffBackend>(
    model_path: &Path,
    config: ModelConfig,
    tokens: &[u32],
    output: &Path,
    device: B::Device,
    max_epochs: usize,
) -> Result<()> {
    let ctx_len = config.ctx_len;

    if tokens.len() < ctx_len + 1 {
        return Err(crate::Error::Burn(format!(
            "Not enough tokens ({}) for context length ({}). Need at least {} tokens.",
            tokens.len(),
            ctx_len,
            ctx_len + 1
        )));
    }

    // Load pre-trained model and wrap in trainer with fresh optimizer
    eprintln!("Loading pre-trained model from {}...", model_path.display());
    let model: Model<B> = Model::load(model_path, &device)?;
    let mut trainer: Trainer<B> = Trainer::from_model(model, config, &device);

    // Scale learning rate linearly with batch size
    let learning_rate = BASE_LEARNING_RATE * (BATCH_SIZE as f64 / BASE_BATCH_SIZE as f64);
    eprintln!(
        "Learning rate: {:.2e} (base {:.2e} scaled for batch size {})",
        learning_rate, BASE_LEARNING_RATE, BATCH_SIZE
    );

    // Split tokens into train and validation sets
    let val_size = ((tokens.len() as f32) * VAL_SPLIT) as usize;
    let train_size = tokens.len() - val_size;
    let train_tokens = &tokens[..train_size];
    let val_tokens = &tokens[train_size..];

    eprintln!(
        "Data split: {} train tokens, {} val tokens ({:.0}% val)",
        train_size,
        val_size,
        VAL_SPLIT * 100.0
    );

    let tokens_per_batch = BATCH_SIZE * ctx_len;
    let num_train_batches = (train_tokens.len() - 1) / tokens_per_batch;
    let num_val_batches = (val_tokens.len() - 1) / tokens_per_batch;

    if num_train_batches == 0 {
        return Err(crate::Error::Burn(format!(
            "Not enough train tokens ({}) for batch size {} x context {}. Need at least {} tokens.",
            train_tokens.len(),
            BATCH_SIZE,
            ctx_len,
            tokens_per_batch + 1
        )));
    }

    if num_val_batches == 0 {
        return Err(crate::Error::Burn(format!(
            "Not enough val tokens ({}) for batch size {} x context {}. Need at least {} tokens.",
            val_tokens.len(),
            BATCH_SIZE,
            ctx_len,
            tokens_per_batch + 1
        )));
    }

    eprintln!(
        "Fine-tuning: {} train batches, {} val batches ({} sequences x {} tokens)",
        num_train_batches, num_val_batches, BATCH_SIZE, ctx_len
    );
    eprintln!(
        "Early stopping: patience={}, min_improvement={} (based on val loss)",
        PATIENCE, MIN_IMPROVEMENT
    );

    let train_token_ids: Vec<i32> = train_tokens.iter().map(|&t| t as i32).collect();
    let val_token_ids: Vec<i32> = val_tokens.iter().map(|&t| t as i32).collect();

    let mut best_loss = f32::MAX;
    let mut epochs_without_improvement = 0;

    for epoch in 0..max_epochs {
        eprintln!("Epoch {} starting...", epoch + 1);
        let epoch_start = Instant::now();
        let mut total_train_loss = 0.0;
        let mut interval_loss = 0.0;
        let mut interval_count = 0;

        for batch_idx in 0..num_train_batches {
            if batch_idx == 0 {
                eprintln!("  batch 1: starting (first batch may be slow)...");
            }

            let batch_start = batch_idx * tokens_per_batch;

            let mut input_data = Vec::with_capacity(tokens_per_batch);
            let mut target_data = Vec::with_capacity(tokens_per_batch);

            for i in 0..tokens_per_batch {
                input_data.push(train_token_ids[batch_start + i]);
                target_data.push(train_token_ids[batch_start + i + 1]);
            }

            let input: Tensor<B, 2, Int> =
                Tensor::from_data(TensorData::new(input_data, [BATCH_SIZE, ctx_len]), &device);
            let target: Tensor<B, 2, Int> =
                Tensor::from_data(TensorData::new(target_data, [BATCH_SIZE, ctx_len]), &device);

            let loss = trainer.train_step(input, target, learning_rate, &device);
            total_train_loss += loss;
            interval_loss += loss;
            interval_count += 1;

            if batch_idx < 3 {
                let elapsed = epoch_start.elapsed().as_secs_f32();
                let eta_secs = elapsed / (batch_idx + 1) as f32 * num_train_batches as f32;
                eprintln!(
                    "  batch {}/{}: loss = {:.4}, eta {}",
                    batch_idx + 1,
                    num_train_batches,
                    loss,
                    format_duration(eta_secs)
                );
            }

            if (batch_idx + 1) % LOG_INTERVAL == 0 {
                let elapsed = epoch_start.elapsed().as_secs_f32();
                let batches_per_sec = (batch_idx + 1) as f32 / elapsed;
                let eta_secs = (num_train_batches - batch_idx - 1) as f32 / batches_per_sec;
                eprintln!(
                    "  batch {}/{}: loss = {:.4}, {:.2} batches/s, eta {}",
                    batch_idx + 1,
                    num_train_batches,
                    interval_loss / interval_count as f32,
                    batches_per_sec,
                    format_duration(eta_secs)
                );
                interval_loss = 0.0;
                interval_count = 0;
            }
        }

        let avg_train_loss = total_train_loss / num_train_batches as f32;

        // Validation loop (no gradients)
        let mut total_val_loss = 0.0;
        for batch_idx in 0..num_val_batches {
            let batch_start = batch_idx * tokens_per_batch;

            let mut input_data = Vec::with_capacity(tokens_per_batch);
            let mut target_data = Vec::with_capacity(tokens_per_batch);

            for i in 0..tokens_per_batch {
                input_data.push(val_token_ids[batch_start + i]);
                target_data.push(val_token_ids[batch_start + i + 1]);
            }

            let input: Tensor<B, 2, Int> =
                Tensor::from_data(TensorData::new(input_data, [BATCH_SIZE, ctx_len]), &device);
            let target: Tensor<B, 2, Int> =
                Tensor::from_data(TensorData::new(target_data, [BATCH_SIZE, ctx_len]), &device);

            let loss = trainer.eval_step(input, target, &device);
            total_val_loss += loss;
        }
        let avg_val_loss = total_val_loss / num_val_batches as f32;

        let elapsed = epoch_start.elapsed();

        let improved = best_loss - avg_val_loss > MIN_IMPROVEMENT;
        let model_path = output.join("model.mpk");
        if improved {
            best_loss = avg_val_loss;
            epochs_without_improvement = 0;
            trainer.save(&model_path)?;
            eprintln!("  saved best model to {}", model_path.display());
        } else {
            epochs_without_improvement += 1;
        }

        eprintln!(
            "Epoch {}: train_loss = {:.4}, val_loss = {:.4}, best_val = {:.4}, patience = {}/{}, time = {:.1}s",
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
            best_loss,
            PATIENCE - epochs_without_improvement,
            PATIENCE,
            elapsed.as_secs_f32()
        );

        if epochs_without_improvement >= PATIENCE {
            eprintln!("Early stopping: no improvement for {} epochs", PATIENCE);
            break;
        }
    }

    // Test generation with instruction-formatted prompt
    eprintln!("Testing generation...");
    let test_prompt = "### Instruction:\nWhat is machine learning?\n### Response:\n";
    let prompt_tokens = test_prompt.tokenize()?;
    let inner_model = trainer.model.clone().valid();
    let mut rng = rand::thread_rng();
    let config = crate::inference::SamplingConfig::default();

    let mut tokens = prompt_tokens.clone();
    for _ in 0..20 {
        let next = crate::generate_next_token(&inner_model, &tokens, &config, &device, &mut rng);
        tokens.push(next);
    }
    let generated = crate::decode(&tokens)?;
    eprintln!("Test: \"{}\" -> \"{}\"", test_prompt, generated);
    eprintln!(
        "Fine-tuning complete. Best model saved to {}",
        output.join("model.mpk").display()
    );

    Ok(())
}

fn format_duration(secs: f32) -> String {
    let secs = secs as u64;
    let hours = secs / 3600;
    let mins = (secs % 3600) / 60;
    let secs = secs % 60;
    if hours > 0 {
        format!("{}h {:02}m {:02}s", hours, mins, secs)
    } else if mins > 0 {
        format!("{}m {:02}s", mins, secs)
    } else {
        format!("{}s", secs)
    }
}
