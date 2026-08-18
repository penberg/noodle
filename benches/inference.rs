//! Inference benchmarks: the forward pass and single-token generation.
//!
//! `forward` measures the model's forward pass in isolation (embeddings, RoPE,
//! attention, FFN, output projection) with logits at every position — the shape
//! training and eval need, and the one [`Model::forward`] still serves.
//!
//! `generate_next_token` measures the whole inference step as the chat loop calls it:
//! a cached forward pass over one new position, reading the last-position logits back
//! to the host, repetition penalty, and top-k/top-p sampling.
//!
//! Both run at a full 256-token context and batch size 1 — the worst case for a chat
//! session, and the shape `chat.rs` converges on once a conversation gets going.

use std::hint::black_box;

use burn::{
    backend::{Cuda, NdArray, Wgpu, cuda::CudaDevice, ndarray::NdArrayDevice, wgpu::WgpuDevice},
    prelude::Backend,
    tensor::{Int, Tensor, TensorData},
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use noodle::{
    Session,
    inference::SamplingConfig,
    model::{KvCache, Model, ModelConfig},
};
use rand::{SeedableRng, rngs::StdRng};

mod common;

use common::{BenchBackend, CTX_LEN, bench_config, fake_tokens};

/// Seed for the sampling RNG, so the tokens drawn — and therefore the work done by
/// top-k/top-p — are identical across runs.
const RNG_SEED: u64 = 0xD00D1E;

fn bench_inference<B: Backend>(c: &mut Criterion, device: &B::Device, backend_name: &str) {
    let config = bench_config();
    let model: Model<B> = Model::new(&config, device);
    let sampling = SamplingConfig::default();

    // `generate_next_token` is the number that matters; the two `decode` benchmarks exist
    // to say where its time goes. Each stops at a different point in the same work:
    //
    //     decode_gpu           forward_cached + sync
    //     decode_gpu_readback  the above, plus copying the logits to the host
    //     generate_next_token  the above, plus repetition penalty and sampling
    //
    // so decode_gpu is the GPU cost, (readback - gpu) is the transfer, and
    // (generate_next_token - readback) is the host-side sampling work. All three end with
    // the same trailing sync, so its cost cancels rather than landing in a difference.
    //
    // One term is not held fixed: the repetition penalty scans the whole conversation, and
    // `generate_next_token` extends that conversation on every iteration, so its cost grows
    // over a run while the decode benchmarks stay put. It is a scan of a few thousand u32s
    // against a vocabulary-sized f32 array -- around 4 ns per token of history, so roughly
    // 8 us at the ~2000-token conversation a default-length run reaches, against a step of
    // some 6.8 ms. Measured across histories from 256 to 16000 tokens the difference stayed
    // inside the run-to-run spread. Read the sampling stage as holding at a conversation of
    // that order rather than as a constant.
    //
    // The first three borrow the model; `generate_next_token` consumes it into a Session,
    // so it has to run last. One model rather than four keeps the benchmark's footprint to
    // a single set of weights.
    bench_forward::<B>(c, &model, &config, device, backend_name);
    bench_decode::<B>(c, &model, device, backend_name, false);
    bench_decode::<B>(c, &model, device, backend_name, true);
    bench_generate::<B>(c, model, &sampling, device, backend_name);
}

/// The forward pass alone: `[1, CTX_LEN] -> [1, CTX_LEN, vocab_size]`.
fn bench_forward<B: Backend>(
    c: &mut Criterion,
    model: &Model<B>,
    config: &ModelConfig,
    device: &B::Device,
    backend_name: &str,
) {
    let tokens: Vec<i32> = fake_tokens(CTX_LEN).into_iter().map(|t| t as i32).collect();
    let input: Tensor<B, 2, Int> = Tensor::from_data(TensorData::new(tokens, [1, CTX_LEN]), device);

    // One untimed pass first: GPU backends compile shaders and allocate buffers on the
    // first call for a given shape, which would otherwise land in Criterion's warm-up
    // and blow up its estimate of how long a sample takes.
    black_box(model.forward(input.clone(), device));
    sync::<B>(device);

    let mut group = c.benchmark_group(format!("inference/{backend_name}"));
    // Throughput in tokens per second across the context.
    group.throughput(Throughput::Elements(CTX_LEN as u64));
    group.bench_function("forward", |b| {
        b.iter(|| {
            let logits = model.forward(black_box(input.clone()), device);
            // GPU backends queue work asynchronously; without a sync we would be timing
            // kernel submission instead of execution.
            sync::<B>(device);
            debug_assert_eq!(logits.dims(), [1, CTX_LEN, config.vocab_size]);
            black_box(logits)
        });
    });
    group.finish();
}

/// One decode step against a warm cache, stopping short of the host readback (`readback`
/// false) or including it (`readback` true).
///
/// This mirrors what `Session` does per token: decode one position at a time, and when the
/// cache fills, drop it and re-prefill half a window. Keeping that eviction here means the
/// three generation benchmarks measure the same underlying work and can be subtracted from
/// one another.
fn bench_decode<B: Backend>(
    c: &mut Criterion,
    model: &Model<B>,
    device: &B::Device,
    backend_name: &str,
    readback: bool,
) {
    let pool = fake_tokens(CTX_LEN);
    let mut cache = model.new_cache();
    let mut fed = 0usize;

    // One step per call, refilling the cache the way a long conversation would.
    let step = |cache: &mut KvCache<B>, fed: &mut usize| {
        if cache.len() + 1 > CTX_LEN {
            cache.clear();
        }
        if cache.is_empty() {
            let window: Vec<i32> = pool[..CTX_LEN / 2].iter().map(|&t| t as i32).collect();
            let input = Tensor::from_data(TensorData::new(window, [1, CTX_LEN / 2]), device);
            return model.forward_cached(input, cache, device);
        }

        let token = pool[*fed % pool.len()] as i32;
        *fed += 1;
        let input = Tensor::from_data(TensorData::new(vec![token], [1, 1]), device);
        model.forward_cached(input, cache, device)
    };

    // Untimed calls first: shader compilation and buffer allocation land on the first call
    // for a given shape, and the cache has to be warm before a decode step is a decode step.
    for _ in 0..4 {
        black_box(step(&mut cache, &mut fed));
    }
    sync::<B>(device);

    let mut group = c.benchmark_group(format!("inference/{backend_name}"));
    group.throughput(Throughput::Elements(1));
    let name = if readback {
        "decode_gpu_readback"
    } else {
        "decode_gpu"
    };
    group.bench_function(name, |b| {
        b.iter(|| {
            let logits = step(&mut cache, &mut fed);
            if readback {
                // Forces the queued work to finish and copies vocab_size floats to the host.
                black_box(logits.into_data().to_vec::<f32>().unwrap());
            } else {
                // No copy, but still wait for the GPU, or we would time submission only.
                black_box(&logits);
            }
            // Every generation benchmark ends with the same sync, whether or not it is
            // redundant, so that its host cost cancels when they are subtracted.
            sync::<B>(device);
        });
    });
    group.finish();
}

/// One full generation step, as `chat` and the training smoke test call it.
///
/// The session is primed with a full context and then generates continuously, so this is
/// steady-state decoding against a warm cache rather than a cold first token. That also
/// means the measurement absorbs the cost of eviction: once the conversation outgrows the
/// context window the session drops half the cache and re-prefills, which by design
/// happens once every `CTX_LEN / 2` tokens. Amortizing that over a run is the point —
/// it's what a long chat actually pays per token.
fn bench_generate<B: Backend>(
    c: &mut Criterion,
    model: Model<B>,
    sampling: &SamplingConfig,
    device: &B::Device,
    backend_name: &str,
) {
    let mut session = Session::new(model, device.clone());
    session.push(&fake_tokens(CTX_LEN));

    // Untimed calls first, for the same reason as in `bench_forward`, and to get past the
    // prefill so that timing starts against a warm cache.
    let mut warmup_rng = StdRng::seed_from_u64(RNG_SEED);
    for _ in 0..3 {
        black_box(session.next_token(sampling, &mut warmup_rng));
    }
    sync::<B>(device);

    let mut group = c.benchmark_group(format!("inference/{backend_name}"));
    // One token out per call: this is the metric that matters for interactive
    // generation (tokens per second).
    group.throughput(Throughput::Elements(1));
    group.bench_function("generate_next_token", |b| {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        b.iter(|| {
            let token = session.next_token(sampling, black_box(&mut rng));
            // Reading the logits back to the host already forces the queued work to
            // finish; this only guards against that ceasing to be true.
            sync::<B>(device);
            black_box(token)
        });
    });
    group.finish();
}

fn sync<B: Backend>(device: &B::Device) {
    B::sync(device).expect("backend sync failed");
}

fn benches(c: &mut Criterion) {
    match BenchBackend::from_env() {
        BenchBackend::Cpu => {
            bench_inference::<NdArray<f32>>(c, &NdArrayDevice::default(), BenchBackend::Cpu.name())
        }
        BenchBackend::Wgpu => {
            bench_inference::<Wgpu<f32, i32>>(c, &WgpuDevice::default(), BenchBackend::Wgpu.name())
        }
        BenchBackend::Cuda => {
            bench_inference::<Cuda<f32, i32>>(c, &CudaDevice::default(), BenchBackend::Cuda.name())
        }
    }
}

criterion_group! {
    name = inference;
    // Ten samples rather than Criterion's default hundred: a full-context forward pass
    // is slow enough on the CPU backend that the default would take minutes per case,
    // and on GPU each sample still batches enough iterations to be stable.
    config = Criterion::default().sample_size(10);
    targets = benches
}
criterion_main!(inference);
