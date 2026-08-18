//! Training benchmarks: one optimizer step, and the validation step it is compared against.
//!
//! `train_step` is the inner loop of `train.rs`: forward pass, cross-entropy loss,
//! backward pass, gradient clipping, and an AdamW update. `eval_step` is the
//! forward-only validation path — benchmarking them together shows what the backward
//! pass and optimizer update actually cost.
//!
//! Both run at the shape training uses: a full 256-token context, batch size 8.
//! Fine-tuning's masked step (`Trainer::train_step_masked`) isn't benchmarked
//! separately: it differs only in passing a pad token to the loss, which is not
//! measurably more work than the unmasked loss it replaces.

use std::{hint::black_box, time::Duration};

use burn::{
    backend::{
        Autodiff, Cuda, NdArray, Wgpu, cuda::CudaDevice, ndarray::NdArrayDevice, wgpu::WgpuDevice,
    },
    prelude::Backend,
    tensor::{Int, Tensor, TensorData, backend::AutodiffBackend},
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use noodle::model::Trainer;

mod common;

use common::{BenchBackend, CTX_LEN, TRAIN_BATCH_SIZE, bench_config, fake_tokens};

/// Learning rate used for the benchmarked steps. The value doesn't affect the cost of
/// a step, but matching `train.rs` keeps the benchmark honest about what it simulates.
const LEARNING_RATE: f64 = 2e-4;

fn bench_training<B: AutodiffBackend>(c: &mut Criterion, device: &B::Device, backend_name: &str) {
    bench_train_step::<B>(c, device, backend_name);
    bench_eval_step::<B>(c, device, backend_name);
}

/// Forward + backward + AdamW update, at the pretraining batch size.
fn bench_train_step<B: AutodiffBackend>(c: &mut Criterion, device: &B::Device, backend_name: &str) {
    let mut trainer: Trainer<B> = Trainer::new(bench_config(), device);
    let (input, target) = batch_tensors::<B>(TRAIN_BATCH_SIZE, device);

    // One untimed step first: it compiles GPU kernels and populates AdamW's moment
    // state, so what Criterion measures is a steady-state step rather than the more
    // expensive first one.
    black_box(trainer.train_step(input.clone(), target.clone(), LEARNING_RATE, device));

    let mut group = c.benchmark_group(format!("training/{backend_name}"));
    group.throughput(Throughput::Elements((TRAIN_BATCH_SIZE * CTX_LEN) as u64));
    group.bench_function("train_step", |b| {
        b.iter(|| {
            let loss = trainer.train_step(
                black_box(input.clone()),
                black_box(target.clone()),
                LEARNING_RATE,
                device,
            );
            // `train_step` reads the loss back to the host, which already forces the
            // queued work to complete, so no extra sync is needed here.
            black_box(loss)
        });
    });
    group.finish();
}

/// Validation: forward pass and loss only, no gradients and no optimizer update.
fn bench_eval_step<B: AutodiffBackend>(c: &mut Criterion, device: &B::Device, backend_name: &str) {
    let trainer: Trainer<B> = Trainer::new(bench_config(), device);
    let (input, target) = batch_tensors::<B>(TRAIN_BATCH_SIZE, device);
    black_box(trainer.eval_step(input.clone(), target.clone(), device));

    let mut group = c.benchmark_group(format!("training/{backend_name}"));
    group.throughput(Throughput::Elements((TRAIN_BATCH_SIZE * CTX_LEN) as u64));
    group.bench_function("eval_step", |b| {
        b.iter(|| {
            let loss =
                trainer.eval_step(black_box(input.clone()), black_box(target.clone()), device);
            black_box(loss)
        });
    });
    group.finish();
}

/// Build one `(input, target)` batch of next-token-prediction data, laid out the same
/// way `train.rs` does it: consecutive tokens, with the target shifted by one.
fn batch_tensors<B: Backend>(
    batch: usize,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    let tokens = fake_tokens(batch * CTX_LEN + 1);
    let input: Vec<i32> = tokens[..batch * CTX_LEN]
        .iter()
        .map(|&t| t as i32)
        .collect();
    let target: Vec<i32> = tokens[1..].iter().map(|&t| t as i32).collect();

    (
        Tensor::from_data(TensorData::new(input, [batch, CTX_LEN]), device),
        Tensor::from_data(TensorData::new(target, [batch, CTX_LEN]), device),
    )
}

fn benches(c: &mut Criterion) {
    match BenchBackend::from_env() {
        BenchBackend::Cpu => bench_training::<Autodiff<NdArray<f32>>>(
            c,
            &NdArrayDevice::default(),
            BenchBackend::Cpu.name(),
        ),
        BenchBackend::Wgpu => bench_training::<Autodiff<Wgpu<f32, i32>>>(
            c,
            &WgpuDevice::default(),
            BenchBackend::Wgpu.name(),
        ),
        BenchBackend::Cuda => bench_training::<Autodiff<Cuda<f32, i32>>>(
            c,
            &CudaDevice::default(),
            BenchBackend::Cuda.name(),
        ),
    }
}

criterion_group! {
    name = training;
    // Training steps are slow enough that Criterion's default 100 samples would take
    // many minutes; the longer measurement window leaves room for the ten samples to
    // actually complete on the CPU backend.
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    targets = benches
}
criterion_main!(training);
