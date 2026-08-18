use burn::{
    prelude::Backend,
    tensor::{Int, Tensor, TensorData},
};
use rand::Rng;

use crate::{
    model::{KvCache, Model},
    tokenizer::Token,
};

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

/// One conversation's generation state: a model, the keys and values it has already
/// computed, and every token seen so far.
///
/// Holding these together is what makes the cache usable. The model and its position
/// tables are immutable and could be shared by any number of conversations; the cache is
/// mutable and belongs to exactly one, because it is the conversation, encoded.
pub struct Session<B: Backend> {
    model: Model<B>,
    cache: KvCache<B>,
    device: B::Device,
    /// Every token of the conversation, prompt and generated alike.
    history: Vec<Token>,
    /// Index into `history` of the oldest token the model still attends to. Everything
    /// before it has been evicted.
    window_start: usize,
}

impl<B: Backend> Session<B> {
    pub fn new(model: Model<B>, device: B::Device) -> Self {
        let cache = model.new_cache();

        Self {
            model,
            cache,
            device,
            history: Vec::new(),
            window_start: 0,
        }
    }

    /// Every token of the conversation so far.
    pub fn history(&self) -> &[Token] {
        &self.history
    }

    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Add tokens to the conversation without running the model over them yet; the next
    /// call to [`Self::next_token`] will.
    pub fn push(&mut self, tokens: &[Token]) {
        self.history.extend_from_slice(tokens);
    }

    /// Drop the most recently generated token from the conversation.
    ///
    /// [`Self::next_token`] appends what it samples, because the next call has to run the
    /// model over it. A caller that stops on a token -- an end-of-sequence marker, say --
    /// does not want it in the history at all: it would become context for the next turn
    /// and would be penalized as a repeat, making the model less willing to stop again.
    ///
    /// This is only valid for a token the model has not yet seen, which is exactly the one
    /// `next_token` just returned; the cache still ends at the token before it, so nothing
    /// has to be undone there.
    ///
    /// Discarding leaves the cache covering the whole conversation, so more tokens have to
    /// be [pushed](Self::push) before generating again -- which is what a caller that
    /// stopped on this token is going to do anyway.
    pub fn discard_last(&mut self) -> Option<Token> {
        assert!(
            self.history.len() > self.window_start + self.cache.len(),
            "the last token has already been through the model and cannot be discarded",
        );

        self.history.pop()
    }

    /// Generate one token, extending the conversation by it.
    pub fn next_token<R: Rng>(&mut self, config: &SamplingConfig, rng: &mut R) -> Token {
        assert!(
            !self.history.is_empty(),
            "cannot generate from an empty conversation",
        );

        let mut logits = self.advance();

        // Penalize against the whole conversation, including tokens the model can no
        // longer see, so that eviction does not make it start repeating itself.
        if config.repetition_penalty != 1.0 {
            apply_repetition_penalty(&mut logits, &self.history, config.repetition_penalty);
        }

        let token = sample_token(&logits, config, rng);
        self.history.push(token);

        token
    }

    /// Bring the cache up to date with `history` and return the logits for its last
    /// position.
    fn advance(&mut self) -> Vec<f32> {
        let ctx_len = self.model.ctx_len();

        // Evict once the conversation outgrows the context window. Dropping half of it at
        // a time keeps this amortized: one rebuild every ctx_len/2 tokens, rather than one
        // per token as would happen if we only ever made room for a single token.
        //
        // Surviving entries keep the rotation they were cached with. That is sound because
        // RoPE attention depends only on the distance between two positions, and every
        // surviving pair is still the distance apart it always was.
        if self.history.len() - self.window_start > ctx_len {
            // Half a window, but never nothing: at ctx_len == 1 the halving rounds to zero
            // and would leave the model with no tokens to run over at all.
            let keep = (ctx_len / 2).max(1);

            self.window_start = self.history.len() - keep;
            self.cache.clear();
        }

        let cached_end = self.window_start + self.cache.len();
        assert!(
            cached_end < self.history.len(),
            "nothing new to run the model over",
        );

        let logits = if self.cache.is_empty() {
            // Prefill: the whole window in one pass, which is far better work for the GPU
            // than the same tokens fed one at a time.
            let window = token_ids(&self.history[self.window_start..], &self.device);
            self.model
                .forward_cached(window, &mut self.cache, &self.device)
        } else {
            // Decode: one token at a time, since appending to a warm cache cannot be
            // batched without a rectangular mask.
            let mut logits = None;
            for i in cached_end..self.history.len() {
                let next = token_ids(&[self.history[i]], &self.device);
                logits = Some(
                    self.model
                        .forward_cached(next, &mut self.cache, &self.device),
                );
            }
            logits.expect("at least one uncached token")
        };

        // Sampling happens on the host in f32 whatever precision the model
        // computes in, so convert rather than assume the backend's float type.
        logits
            .reshape([self.model.vocab_size()])
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap()
    }
}

/// Pack tokens into a `[1, len]` tensor.
fn token_ids<B: Backend>(tokens: &[Token], device: &B::Device) -> Tensor<B, 2, Int> {
    let data: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();

    Tensor::from_data(TensorData::new(data, [1, tokens.len()]), device)
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

    // Dividing by a positive temperature is monotonic, so it cannot reorder anything: rank
    // on the raw logits and scale only the handful that survive top-k. Scaling the whole
    // vocabulary first meant 50281 divisions per token to use 40 of them.
    let top_k = config.top_k.clamp(1, logits.len());
    let by_logit_desc = |&a: &u32, &b: &u32| logits[b as usize].total_cmp(&logits[a as usize]);

    // Partition the k largest to the front in linear time, then order just those. Sorting
    // the whole vocabulary to keep the top 40 was the most expensive thing on the host
    // side of a decode step.
    let mut indices: Vec<u32> = (0..logits.len() as u32).collect();
    indices.select_nth_unstable_by(top_k - 1, by_logit_desc);
    indices.truncate(top_k);
    indices.sort_unstable_by(by_logit_desc);

    // Compute softmax over filtered tokens
    let scale = |i: u32| logits[i as usize] / config.temperature;
    let max_logit = scale(indices[0]);
    let mut probs: Vec<(usize, f32)> = indices
        .iter()
        .map(|&i| (i as usize, (scale(i) - max_logit).exp()))
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    use crate::model::{Model, ModelConfig};
    use burn::backend::{NdArray, ndarray::NdArrayDevice};

    /// A caller that stops on a token must be able to keep it out of the conversation, or
    /// it becomes context for the next turn and counts against the repetition penalty.
    #[test]
    fn discarded_token_leaves_no_trace() {
        type B = NdArray<f32>;

        let device = NdArrayDevice::default();
        let config = ModelConfig {
            preset: None,
            n_layers: 2,
            emb_dim: 32,
            n_heads: 2,
            head_dim: 16,
            hidden_dim: 128,
            ctx_len: 16,
            vocab_size: 64,
        };
        let model = Model::<B>::new(&config, &device);

        let mut session = Session::new(model, device);
        session.push(&[3, 14, 15]);

        let mut rng = StdRng::seed_from_u64(7);
        let generated = session.next_token(&SamplingConfig::default(), &mut rng);
        assert_eq!(session.history().len(), 4);

        assert_eq!(session.discard_last(), Some(generated));
        assert_eq!(session.history(), &[3, 14, 15]);

        // And the session still works for the turn that follows, which is the flow `chat`
        // uses: drop the token it stopped on, take more input, keep going.
        session.push(&[9, 26]);
        let next = session.next_token(&SamplingConfig::default(), &mut rng);
        assert_eq!(session.history(), &[3, 14, 15, 9, 26, next]);
    }

    /// A one-token context window is degenerate but was valid before the cache existed:
    /// eviction has to keep a token to run the model over rather than emptying the window.
    #[test]
    fn single_token_context_keeps_generating() {
        type B = NdArray<f32>;

        let device = NdArrayDevice::default();
        let config = ModelConfig {
            preset: None,
            n_layers: 2,
            emb_dim: 32,
            n_heads: 2,
            head_dim: 16,
            hidden_dim: 128,
            ctx_len: 1,
            vocab_size: 64,
        };
        let model = Model::<B>::new(&config, &device);

        let mut session = Session::new(model, device);
        session.push(&[7]);

        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..4 {
            session.next_token(&SamplingConfig::default(), &mut rng);
        }

        assert_eq!(session.history().len(), 5);
    }

    /// Deterministic pseudo-random logits, so the test exercises a realistic spread rather
    /// than a sorted or constant vector.
    fn fake_logits(n: usize) -> Vec<f32> {
        let mut state: u32 = 0x1234_5678;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 20.0 - 10.0
            })
            .collect()
    }

    /// Selecting the top k by partition must pick the same set, in the same order, as
    /// sorting everything and taking the first k. This is the property the fast path
    /// replaced a full sort with.
    #[test]
    fn top_k_selection_matches_full_sort() {
        let logits = fake_logits(50281);
        let k = 40;

        let mut expected: Vec<u32> = (0..logits.len() as u32).collect();
        expected.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
        expected.truncate(k);

        let by_logit_desc = |&a: &u32, &b: &u32| logits[b as usize].total_cmp(&logits[a as usize]);
        let mut actual: Vec<u32> = (0..logits.len() as u32).collect();
        actual.select_nth_unstable_by(k - 1, by_logit_desc);
        actual.truncate(k);
        actual.sort_unstable_by(by_logit_desc);

        let values = |v: &[u32]| v.iter().map(|&i| logits[i as usize]).collect::<Vec<_>>();
        assert_eq!(values(&expected), values(&actual));
    }

    /// A zero or negative temperature takes the greedy path, which must return the argmax.
    #[test]
    fn greedy_sampling_returns_argmax() {
        let mut logits = fake_logits(1024);
        logits[777] = 100.0;

        let config = SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(1);

        assert_eq!(sample_token(&logits, &config, &mut rng), 777);
    }

    /// Every token the sampler can return has to come from the top-k set, whatever the
    /// draw: top_k = 1 pins it to the single most likely token.
    #[test]
    fn sampling_respects_top_k() {
        let mut logits = fake_logits(1024);
        logits[321] = 50.0;

        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 1,
            ..SamplingConfig::default()
        };

        for seed in 0..16 {
            let mut rng = StdRng::seed_from_u64(seed);
            assert_eq!(sample_token(&logits, &config, &mut rng), 321);
        }
    }
}
