use std::io::{self, BufRead, Write};
use std::path::Path;

use burn::backend::cuda::CudaDevice;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Cuda, NdArray, Wgpu};
use burn::prelude::Backend;

use noodle::Tokenizer;
use noodle::inference::SamplingConfig;
use noodle::model::Model;

const EOS_TOKEN: noodle::Token = 50256;

pub fn chat(model_path: &Path, backend: noodle::Backend) -> noodle::Result<()> {
    match backend {
        noodle::Backend::Wgpu => {
            let device = WgpuDevice::default();
            eprintln!("Using wgpu device: {:?}", device);
            chat_loop::<Wgpu<f32, i32>>(model_path, device)
        }
        noodle::Backend::Cuda => {
            let device = CudaDevice::default();
            eprintln!("Using CUDA device: {:?}", device);
            chat_loop::<Cuda<f32, i32>>(model_path, device)
        }
        noodle::Backend::Cpu => {
            let device = NdArrayDevice::default();
            eprintln!("Using CPU device: {:?}", device);
            chat_loop::<NdArray<f32>>(model_path, device)
        }
    }
}

fn chat_loop<B: Backend>(model_path: &Path, device: B::Device) -> noodle::Result<()> {
    let model = Model::<B>::load(model_path, &device)?;
    let tokenizer = Tokenizer::new()?;

    println!();
    println!(" ~(°◡°)~  Noodle");
    println!();
    println!("I am ready to chat! Type your message and press Enter.");
    println!();

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut rng = rand::thread_rng();

    let config = SamplingConfig::default();
    let max_tokens = 100;

    let mut history: Vec<noodle::Token> = Vec::new();

    loop {
        print!("> ");
        stdout.flush()?;

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            break; // EOF
        }

        let input_tokens = tokenizer.encode(input.trim());
        history.extend_from_slice(&input_tokens);

        // Generate tokens one at a time, streaming output
        for _ in 0..max_tokens {
            let next_token =
                noodle::generate_next_token(&model, &history, &config, &device, &mut rng);

            if next_token == EOS_TOKEN {
                break;
            }

            history.push(next_token);

            // Decode and print just this token
            let token_text = tokenizer.decode(&[next_token])?;
            print!("{}", token_text);
            stdout.flush()?;
        }
        println!();
    }

    Ok(())
}
