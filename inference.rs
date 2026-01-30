use burn::{
    prelude::Backend,
    tensor::{Int, Tensor, TensorData},
};
use rand::Rng;

use crate::{model::Model, tokenizer::Token};

/// Sampling configuration for text generation
#[derive(Clone, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.1,
        }
    }
}

/// Generate the next token given current context
pub fn generate_next_token<B: Backend, R: Rng>(
    model: &Model<B>,
    tokens: &[Token],
    config: &SamplingConfig,
    device: &B::Device,
    rng: &mut R,
) -> Token {
    let ctx_len = model.ctx_len();

    // Truncate to context length if needed
    let context: &[Token] = if tokens.len() > ctx_len {
        &tokens[tokens.len() - ctx_len..]
    } else {
        tokens
    };

    // Forward pass
    let seq_len = context.len();
    let token_data: Vec<i32> = context.iter().map(|&t| t as i32).collect();
    let token_ids: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(token_data, [1, seq_len]), device);

    let logits_tensor = model.forward(token_ids, device);

    // Extract last position logits
    let vocab_size = model.vocab_size();
    let last_logits = logits_tensor.slice([0..1, (seq_len - 1)..seq_len, 0..vocab_size]);
    let last_logits = last_logits.reshape([vocab_size]);
    let mut logits = last_logits.into_data().to_vec::<f32>().unwrap();

    // Apply repetition penalty to tokens that have already appeared
    if config.repetition_penalty != 1.0 {
        apply_repetition_penalty(&mut logits, tokens, config.repetition_penalty);
    }

    // Sample next token
    sample_token(&logits, config, rng)
}

/// Apply repetition penalty to logits for tokens that have appeared in context
fn apply_repetition_penalty(logits: &mut [f32], tokens: &[Token], penalty: f32) {
    for &token in tokens {
        let idx = token as usize;
        if idx < logits.len() {
            // If logit is positive, divide by penalty; if negative, multiply
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

fn sample_token<R: Rng>(logits: &[f32], config: &SamplingConfig, rng: &mut R) -> Token {
    if config.temperature <= 0.0 {
        // Greedy: pick highest logit
        return argmax(logits) as Token;
    }

    // Apply temperature
    let scaled: Vec<f32> = logits.iter().map(|&x| x / config.temperature).collect();

    // Get indices sorted by logit value (descending)
    let mut indices: Vec<usize> = (0..scaled.len()).collect();
    indices.sort_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap());

    // Apply top-k: keep only top k tokens
    let top_k = config.top_k.min(indices.len());
    let indices: Vec<usize> = indices.into_iter().take(top_k).collect();

    // Compute softmax over filtered tokens
    let max_logit = scaled[indices[0]];
    let mut probs: Vec<(usize, f32)> = indices
        .iter()
        .map(|&i| (i, (scaled[i] - max_logit).exp()))
        .collect();

    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    for (_, p) in &mut probs {
        *p /= sum;
    }

    // Apply top-p (nucleus sampling): keep tokens until cumulative prob >= top_p
    let mut cumsum = 0.0;
    let mut nucleus: Vec<(usize, f32)> = Vec::new();
    for (idx, prob) in probs {
        cumsum += prob;
        nucleus.push((idx, prob));
        if cumsum >= config.top_p {
            break;
        }
    }

    // Renormalize nucleus probabilities
    let nucleus_sum: f32 = nucleus.iter().map(|(_, p)| p).sum();
    for (_, p) in &mut nucleus {
        *p /= nucleus_sum;
    }

    // Sample from nucleus
    let r: f32 = rng.r#gen();
    let mut cumsum = 0.0;
    for (idx, prob) in &nucleus {
        cumsum += prob;
        if r < cumsum {
            return *idx as Token;
        }
    }

    // Fallback to first token in nucleus
    nucleus[0].0 as Token
}

fn argmax(logits: &[f32]) -> usize {
    let mut max_idx = 0;
    let mut max_val = logits[0];
    for (i, &val) in logits.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    max_idx
}
