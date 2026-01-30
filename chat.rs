use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{NdArray, Wgpu};
use burn::prelude::Backend;

use noodle::Tokenize;
use noodle::inference::SamplingConfig;
use noodle::model::Model;

pub fn chat(model_path: &PathBuf, use_gpu: bool) -> noodle::Result<()> {
    if use_gpu {
        let device = WgpuDevice::default();
        eprintln!("Using GPU device: {:?}", device);
        chat_loop::<Wgpu<f32, i32>>(model_path, device)
    } else {
        let device = NdArrayDevice::default();
        eprintln!("Using CPU device: {:?}", device);
        chat_loop::<NdArray<f32>>(model_path, device)
    }
}

fn chat_loop<B: Backend>(model_path: &PathBuf, device: B::Device) -> noodle::Result<()> {
    let model = Model::<B>::load(model_path, &device)?;

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

    loop {
        print!("> ");
        stdout.flush()?;

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            break; // EOF
        }

        let mut tokens = input.trim().tokenize()?;

        // Generate tokens one at a time, streaming output
        for _ in 0..max_tokens {
            let next_token =
                noodle::generate_next_token(&model, &tokens, &config, &device, &mut rng);

            tokens.push(next_token);

            // Decode and print just this token
            let token_text = noodle::decode(&[next_token])?;
            print!("{}", token_text);
            stdout.flush()?;

            // TODO: stop on EOS token
        }
        println!();
    }

    Ok(())
}
