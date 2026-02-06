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
    tokenizer::{Token, Tokenizer},
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

// EOS token for p50k_base tokenizer
const EOS_TOKEN: Token = 50256;

// PAD token ID used to mask loss (must not collide with real vocab)
const PAD_ID: i32 = 50257;

/// A parsed instruction-tuning example: (prompt_tokens, response_tokens)
type Example = (Vec<Token>, Vec<Token>);

/// Parse a dolly-format instruction file into (prompt, response) token pairs.
///
/// The file format has examples separated by `<|endoftext|>`, each containing:
/// ```text
/// ### Instruction:
/// <instruction text>
///
/// ### Input:        (optional)
/// <context text>
///
/// ### Response:
/// <response text>
/// ```
fn parse_instruction_file(path: &Path) -> Result<Vec<Example>> {
    let text = std::fs::read_to_string(path)?;
    let tokenizer = Tokenizer::new()?;
    let mut examples = Vec::new();

    for block in text.split("<|endoftext|>") {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }

        // Extract sections
        let instruction = extract_section(block, "### Instruction:");
        let input = extract_section(block, "### Input:");
        let response = extract_section(block, "### Response:");

        let instruction = match instruction {
            Some(s) if !s.is_empty() => s,
            _ => continue,
        };
        let response = match response {
            Some(s) if !s.is_empty() => s,
            _ => continue,
        };

        // Build prompt: instruction + optional input/context
        let prompt = match input {
            Some(ctx) if !ctx.is_empty() => format!("{}\n{}", instruction, ctx),
            _ => instruction.to_string(),
        };

        let prompt_tokens = tokenizer.encode(&prompt);
        let response_tokens = tokenizer.encode(response);

        examples.push((prompt_tokens, response_tokens));
    }

    Ok(examples)
}

/// Extract the text content after a section header (e.g. "### Instruction:").
/// Returns the text between this header and the next "###" header (or end of block).
fn extract_section<'a>(block: &'a str, header: &str) -> Option<&'a str> {
    let start = block.find(header)?;
    let after_header = &block[start + header.len()..];

    // Find the next section header or end of block
    let end = after_header.find("\n### ").unwrap_or(after_header.len());
    let content = after_header[..end].trim();

    Some(content)
}

/// Prepare a single example into input/target arrays of length ctx_len.
///
/// Layout: `[prompt_tokens... response_tokens... EOS PAD...]`
/// Target:  positions under prompt are set to PAD_ID (masked out),
///          positions under response+EOS are the next token,
///          positions under padding are PAD_ID (masked out).
fn prepare_example(
    prompt_tokens: &[Token],
    response_tokens: &[Token],
    ctx_len: usize,
) -> (Vec<i32>, Vec<i32>) {
    // Build full sequence: prompt + response + EOS
    let mut full_seq: Vec<Token> =
        Vec::with_capacity(prompt_tokens.len() + response_tokens.len() + 1);
    full_seq.extend_from_slice(prompt_tokens);
    full_seq.extend_from_slice(response_tokens);
    full_seq.push(EOS_TOKEN);

    // If too long, truncate the prompt (keep response intact)
    let prompt_len = if full_seq.len() > ctx_len + 1 {
        let excess = full_seq.len() - (ctx_len + 1);
        let new_prompt_len = prompt_tokens.len().saturating_sub(excess);
        if new_prompt_len == 0 {
            // Response itself is too long, truncate from end
            full_seq.truncate(ctx_len + 1);
            0
        } else {
            // Remove tokens from the start of the prompt
            full_seq.drain(0..excess);
            new_prompt_len
        }
    } else {
        prompt_tokens.len()
    };

    // Now full_seq.len() <= ctx_len + 1
    // input = full_seq[0..ctx_len], target = full_seq[1..ctx_len+1]
    // Pad to ctx_len + 1 if shorter
    let seq_len = full_seq.len();
    let mut input = Vec::with_capacity(ctx_len);
    let mut target = Vec::with_capacity(ctx_len);

    for i in 0..ctx_len {
        // Input token
        if i < seq_len - 1 {
            input.push(full_seq[i] as i32);
        } else {
            input.push(PAD_ID); // padding in input (won't matter since target is masked)
        }

        // Target token (shifted by 1)
        if i + 1 < seq_len {
            if i < prompt_len.saturating_sub(1) {
                // This target predicts a prompt token — mask it
                target.push(PAD_ID);
            } else {
                target.push(full_seq[i + 1] as i32);
            }
        } else {
            // Padding position — mask it
            target.push(PAD_ID);
        }
    }

    (input, target)
}

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

    // Parse instruction file into (prompt, response) pairs
    let examples = parse_instruction_file(input)?;
    eprintln!("Parsed {} instruction examples", examples.len());

    if examples.is_empty() {
        return Err(crate::Error::Burn(
            "No valid examples found in input file".to_string(),
        ));
    }

    // Log first few examples
    for (i, (prompt_tokens, response_tokens)) in examples.iter().take(3).enumerate() {
        eprintln!(
            "  example {}: {} prompt tokens, {} response tokens",
            i + 1,
            prompt_tokens.len(),
            response_tokens.len()
        );
    }

    match backend {
        crate::Backend::Wgpu => {
            let device = WgpuDevice::default();
            eprintln!("Using wgpu device: {:?}", device);
            finetune_loop::<Autodiff<Wgpu<f32, i32>>>(
                model_path, config, &examples, output, device, max_epochs,
            )
        }
        crate::Backend::Cuda => {
            let device = CudaDevice::default();
            eprintln!("Using CUDA device: {:?}", device);
            finetune_loop::<Autodiff<Cuda<f32, i32>>>(
                model_path, config, &examples, output, device, max_epochs,
            )
        }
        crate::Backend::Cpu => {
            let device = NdArrayDevice::default();
            eprintln!("Using CPU device: {:?}", device);
            finetune_loop::<Autodiff<NdArray<f32>>>(
                model_path, config, &examples, output, device, max_epochs,
            )
        }
    }
}

fn finetune_loop<B: AutodiffBackend>(
    model_path: &Path,
    config: ModelConfig,
    examples: &[Example],
    output: &Path,
    device: B::Device,
    max_epochs: usize,
) -> Result<()> {
    let ctx_len = config.ctx_len;

    if examples.len() < BATCH_SIZE {
        return Err(crate::Error::Burn(format!(
            "Not enough examples ({}) for batch size ({}). Need at least {} examples.",
            examples.len(),
            BATCH_SIZE,
            BATCH_SIZE,
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

    // Split examples into train and validation sets
    let val_size = ((examples.len() as f32) * VAL_SPLIT).max(1.0) as usize;
    let train_size = examples.len() - val_size;
    let train_examples = &examples[..train_size];
    let val_examples = &examples[train_size..];

    eprintln!(
        "Data split: {} train examples, {} val examples ({:.0}% val)",
        train_size,
        val_size,
        VAL_SPLIT * 100.0
    );

    let num_train_batches = train_examples.len() / BATCH_SIZE;
    let num_val_batches = val_examples.len() / BATCH_SIZE;

    if num_train_batches == 0 {
        return Err(crate::Error::Burn(format!(
            "Not enough train examples ({}) for batch size {}.",
            train_examples.len(),
            BATCH_SIZE,
        )));
    }

    if num_val_batches == 0 {
        return Err(crate::Error::Burn(format!(
            "Not enough val examples ({}) for batch size {}.",
            val_examples.len(),
            BATCH_SIZE,
        )));
    }

    eprintln!(
        "Fine-tuning: {} train batches, {} val batches ({} examples/batch, ctx_len={})",
        num_train_batches, num_val_batches, BATCH_SIZE, ctx_len
    );
    eprintln!(
        "Early stopping: patience={}, min_improvement={} (based on val loss)",
        PATIENCE, MIN_IMPROVEMENT
    );

    let pad_id = PAD_ID as usize;

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

            let batch_start = batch_idx * BATCH_SIZE;

            let mut input_data = Vec::with_capacity(BATCH_SIZE * ctx_len);
            let mut target_data = Vec::with_capacity(BATCH_SIZE * ctx_len);

            for i in 0..BATCH_SIZE {
                let (prompt_tokens, response_tokens) = &train_examples[batch_start + i];
                let (inp, tgt) = prepare_example(prompt_tokens, response_tokens, ctx_len);
                input_data.extend(inp);
                target_data.extend(tgt);
            }

            let input: Tensor<B, 2, Int> =
                Tensor::from_data(TensorData::new(input_data, [BATCH_SIZE, ctx_len]), &device);
            let target: Tensor<B, 2, Int> =
                Tensor::from_data(TensorData::new(target_data, [BATCH_SIZE, ctx_len]), &device);

            let loss = trainer.train_step_masked(input, target, learning_rate, pad_id, &device);
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
            let batch_start = batch_idx * BATCH_SIZE;

            let mut input_data = Vec::with_capacity(BATCH_SIZE * ctx_len);
            let mut target_data = Vec::with_capacity(BATCH_SIZE * ctx_len);

            for i in 0..BATCH_SIZE {
                let (prompt_tokens, response_tokens) = &val_examples[batch_start + i];
                let (inp, tgt) = prepare_example(prompt_tokens, response_tokens, ctx_len);
                input_data.extend(inp);
                target_data.extend(tgt);
            }

            let input: Tensor<B, 2, Int> =
                Tensor::from_data(TensorData::new(input_data, [BATCH_SIZE, ctx_len]), &device);
            let target: Tensor<B, 2, Int> =
                Tensor::from_data(TensorData::new(target_data, [BATCH_SIZE, ctx_len]), &device);

            let loss = trainer.eval_step_masked(input, target, pad_id, &device);
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

    // Test generation with a raw question (matching inference format)
    eprintln!("Testing generation...");
    let test_prompt = "What is machine learning?";
    let tokenizer = Tokenizer::new()?;
    let prompt_tokens = tokenizer.encode(test_prompt);
    let inner_model = trainer.model.clone().valid();
    let mut rng = rand::thread_rng();
    let sampling_config = crate::inference::SamplingConfig::default();

    let mut tokens = prompt_tokens.clone();
    for _ in 0..50 {
        let next =
            crate::generate_next_token(&inner_model, &tokens, &sampling_config, &device, &mut rng);
        if next == EOS_TOKEN {
            break;
        }
        tokens.push(next);
    }
    let generated = tokenizer.decode(&tokens)?;
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
