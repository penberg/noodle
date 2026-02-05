use std::{path::Path, time::Instant};

use burn::{
    backend::{
        Autodiff, Cuda, NdArray, Wgpu, cuda::CudaDevice, ndarray::NdArrayDevice, wgpu::WgpuDevice,
    },
    module::AutodiffModule,
    tensor::{Int, Tensor, TensorData, backend::AutodiffBackend},
};
use serde_json;

use crate::{
    Result,
    model::{ModelConfig, Trainer},
    tokenizer::Tokenize,
};

// Canonical Noodle model hyperparameters (sized for small datasets ~300K-1M tokens)
const LAYERS: usize = 4;
const D_MODEL: usize = 256;
const HEADS: usize = 4;
const CTX_LEN: usize = 256;
const VOCAB_SIZE: usize = 50281; // p50k_base tokenizer

// Training hyperparameters
const BATCH_SIZE: usize = 8;
const LOG_INTERVAL: usize = 10;
const BASE_LEARNING_RATE: f64 = 1e-4;
const BASE_BATCH_SIZE: usize = 4; // LR is scaled linearly relative to this

// Step-based evaluation: validate every EVAL_INTERVAL training steps
const EVAL_INTERVAL: usize = 200;
// Early stopping: stop if validation loss doesn't improve by MIN_IMPROVEMENT for PATIENCE evals
const PATIENCE: usize = 10;
const MIN_IMPROVEMENT: f32 = 0.01;

// Train/validation split ratio (90% train, 10% validation)
const VAL_SPLIT: f32 = 0.1;

pub fn train(
    input: &Path,
    output: &Path,
    backend: crate::Backend,
    max_epochs: usize,
) -> Result<()> {
    #[cfg(debug_assertions)]
    eprintln!("Warning: running in debug mode, use --release for faster training");

    // Log available GPU adapters for wgpu backend
    if matches!(backend, crate::Backend::Wgpu) {
        log_gpu_adapters();
    }

    // Create output directory and save config before training
    std::fs::create_dir_all(output)?;
    let config = ModelConfig::new(LAYERS, D_MODEL, HEADS, CTX_LEN, VOCAB_SIZE);
    let config_path = output.join("model.json");
    let config_json =
        serde_json::to_string_pretty(&config).map_err(|e| crate::Error::Burn(e.to_string()))?;
    std::fs::write(&config_path, config_json)?;
    eprintln!("Saved config to {}", config_path.display());

    let tokens = input.tokenize()?;
    eprintln!("Loaded {} tokens", tokens.len());

    match backend {
        crate::Backend::Wgpu => {
            let device = WgpuDevice::default();
            eprintln!("Using wgpu device: {:?}", device);
            train_loop::<Autodiff<Wgpu<f32, i32>>>(config, &tokens, output, device, max_epochs)
        }
        crate::Backend::Cuda => {
            let device = CudaDevice::default();
            eprintln!("Using CUDA device: {:?}", device);
            train_loop::<Autodiff<Cuda<f32, i32>>>(config, &tokens, output, device, max_epochs)
        }
        crate::Backend::Cpu => {
            let device = NdArrayDevice::default();
            eprintln!("Using CPU device: {:?}", device);
            train_loop::<Autodiff<NdArray<f32>>>(config, &tokens, output, device, max_epochs)
        }
    }
}

fn train_loop<B: AutodiffBackend>(
    config: ModelConfig,
    tokens: &[u32],
    output: &Path,
    device: B::Device,
    max_epochs: usize,
) -> Result<()> {
    let ctx_len = config.ctx_len;

    // Validate we have enough tokens for at least one training example
    // We need ctx_len tokens for input and 1 more for the target (next token prediction)
    if tokens.len() < ctx_len + 1 {
        return Err(crate::Error::Burn(format!(
            "Not enough tokens ({}) for context length ({}). Need at least {} tokens.",
            tokens.len(),
            ctx_len,
            ctx_len + 1
        )));
    }

    let mut trainer: Trainer<B> = Trainer::new(config, &device);

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

    // Following nanochat's approach: non-overlapping consecutive chunks
    // Each batch needs BATCH_SIZE * ctx_len tokens for input, plus 1 more for the final target
    // We stream through tokens, advancing by BATCH_SIZE * ctx_len per batch
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

    // Clamp eval_interval for tiny datasets: degrade gracefully to once-per-epoch
    let eval_interval = EVAL_INTERVAL.min(num_train_batches);

    eprintln!(
        "Training: {} train batches, {} val batches ({} sequences x {} tokens)",
        num_train_batches, num_val_batches, BATCH_SIZE, ctx_len
    );
    eprintln!(
        "Eval every {} steps, early stopping: patience={}, min_improvement={} (based on val loss)",
        eval_interval, PATIENCE, MIN_IMPROVEMENT
    );

    // Pre-convert all tokens to i32 for easier batching
    let train_token_ids: Vec<i32> = train_tokens.iter().map(|&t| t as i32).collect();
    let val_token_ids: Vec<i32> = val_tokens.iter().map(|&t| t as i32).collect();

    // Early stopping state
    let mut best_loss = f32::MAX;
    let mut evals_without_improvement = 0;
    let mut global_step: usize = 0;
    let mut stopped_early = false;
    let model_path = output.join("model.mpk");

    for epoch in 0..max_epochs {
        eprintln!("Epoch {} starting...", epoch + 1);
        let epoch_start = Instant::now();
        let mut total_train_loss = 0.0;
        let mut interval_loss = 0.0;
        let mut interval_count = 0;
        let mut last_eval_step = global_step;

        // Training loop
        for batch_idx in 0..num_train_batches {
            if batch_idx == 0 {
                eprintln!("  batch 1: starting (first batch may be slow)...");
            }

            // Build batch tensors using consecutive non-overlapping chunks (like nanochat)
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
            global_step += 1;

            // Log first 3 batches to estimate speed
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

            // Step-based evaluation
            if global_step.is_multiple_of(eval_interval) {
                last_eval_step = global_step;
                let avg_val_loss = validate(
                    &trainer,
                    &val_token_ids,
                    num_val_batches,
                    tokens_per_batch,
                    ctx_len,
                    &device,
                );

                let improved = best_loss - avg_val_loss > MIN_IMPROVEMENT;
                if improved {
                    best_loss = avg_val_loss;
                    evals_without_improvement = 0;
                    trainer.save(&model_path)?;
                    eprintln!(
                        "  [step {}] val_loss = {:.4} (new best), saved model",
                        global_step, avg_val_loss
                    );
                } else {
                    evals_without_improvement += 1;
                    eprintln!(
                        "  [step {}] val_loss = {:.4}, best_val = {:.4}, patience = {}/{}",
                        global_step,
                        avg_val_loss,
                        best_loss,
                        PATIENCE - evals_without_improvement,
                        PATIENCE
                    );
                }

                if evals_without_improvement >= PATIENCE {
                    eprintln!(
                        "Early stopping at step {}: no improvement for {} evals",
                        global_step, PATIENCE
                    );
                    stopped_early = true;
                    break;
                }
            }
        }

        if stopped_early {
            break;
        }

        let avg_train_loss = total_train_loss / num_train_batches as f32;

        // Force validation at epoch boundary if last batch didn't already trigger one
        let avg_val_loss = if last_eval_step == global_step {
            // Already ran validation on the last step of this epoch
            best_loss // Use best_loss for display; actual val was already logged
        } else {
            let val_loss = validate(
                &trainer,
                &val_token_ids,
                num_val_batches,
                tokens_per_batch,
                ctx_len,
                &device,
            );

            let improved = best_loss - val_loss > MIN_IMPROVEMENT;
            if improved {
                best_loss = val_loss;
                evals_without_improvement = 0;
                trainer.save(&model_path)?;
                eprintln!("  saved best model to {}", model_path.display());
            } else {
                evals_without_improvement += 1;
            }

            val_loss
        };

        let elapsed = epoch_start.elapsed();
        eprintln!(
            "Epoch {}: train_loss = {:.4}, val_loss = {:.4}, best_val = {:.4}, patience = {}/{}, time = {:.1}s",
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
            best_loss,
            PATIENCE - evals_without_improvement,
            PATIENCE,
            elapsed.as_secs_f32()
        );

        // Early stopping check at epoch boundary
        if evals_without_improvement >= PATIENCE {
            eprintln!(
                "Early stopping at epoch {}: no improvement for {} evals",
                epoch + 1,
                PATIENCE
            );
            break;
        }
    }

    // Test generation with final model state (note: best model already saved)
    eprintln!("Testing generation...");
    let test_prompt = "Once upon a time";
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
        "Training complete. Best model saved to {}",
        output.join("model.mpk").display()
    );

    Ok(())
}

fn validate<B: AutodiffBackend>(
    trainer: &Trainer<B>,
    val_token_ids: &[i32],
    num_val_batches: usize,
    tokens_per_batch: usize,
    ctx_len: usize,
    device: &B::Device,
) -> f32 {
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
            Tensor::from_data(TensorData::new(input_data, [BATCH_SIZE, ctx_len]), device);
        let target: Tensor<B, 2, Int> =
            Tensor::from_data(TensorData::new(target_data, [BATCH_SIZE, ctx_len]), device);

        let loss = trainer.eval_step(input, target, device);
        total_val_loss += loss;
    }
    total_val_loss / num_val_batches as f32
}

fn log_gpu_adapters() {
    use wgpu::{Backends, Instance, InstanceDescriptor};

    let instance = Instance::new(&InstanceDescriptor {
        backends: Backends::all(),
        ..Default::default()
    });

    let adapters = instance.enumerate_adapters(Backends::all());

    eprintln!("Available GPU adapters ({}):", adapters.len());
    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        eprintln!(
            "  [{}] {} ({:?}, {:?})",
            i, info.name, info.device_type, info.backend
        );
    }

    if adapters.is_empty() {
        eprintln!("  WARNING: No GPU adapters found!");
    }
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
