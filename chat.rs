use std::io::{self, BufRead, Write};
use std::path::Path;

use burn::backend::cuda::CudaDevice;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Cuda, NdArray, Wgpu};
use burn::prelude::Backend;

use noodle::Session;
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

    // The session owns the key/value cache, so it persists across turns: each generated
    // token costs one position of work rather than a pass over the whole conversation.
    let mut session = Session::new(model, device);

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

        session.push(&tokenizer.encode(input.trim()));

        // Nothing said yet and nothing to continue from
        if session.is_empty() {
            continue;
        }

        // Generate tokens one at a time, streaming output. A token is a byte
        // sequence, not a character sequence: a multi-byte character can be
        // split across tokens, so bytes are buffered until they decode.
        let mut pending: Vec<u8> = Vec::new();
        for _ in 0..max_tokens {
            let next_token = session.next_token(&config, &mut rng);

            if next_token == EOS_TOKEN {
                // Keep it out of the conversation: it would be context for the next turn
                // and would count against the repetition penalty.
                session.discard_last();
                break;
            }

            pending.extend_from_slice(&tokenizer.decode_bytes(&[next_token]));
            flush_utf8(&mut pending, &mut stdout)?;
            stdout.flush()?;
        }
        // Generation stopped mid-character; show what remains rather than drop it.
        if !pending.is_empty() {
            print!("{}", String::from_utf8_lossy(&pending));
        }
        println!();
    }

    Ok(())
}

/// Write the longest decodable prefix of `pending` to `out`, leaving only a
/// trailing incomplete UTF-8 sequence (if any) behind for the next token to
/// complete. Invalid bytes come out as U+FFFD so one bad token cannot wedge
/// the stream.
fn flush_utf8(pending: &mut Vec<u8>, out: &mut impl Write) -> io::Result<()> {
    loop {
        match std::str::from_utf8(pending) {
            Ok(text) => {
                out.write_all(text.as_bytes())?;
                pending.clear();
                return Ok(());
            }
            Err(err) => {
                out.write_all(&pending[..err.valid_up_to()])?;
                match err.error_len() {
                    Some(bad) => {
                        out.write_all("\u{FFFD}".as_bytes())?;
                        pending.drain(..err.valid_up_to() + bad);
                    }
                    None => {
                        pending.drain(..err.valid_up_to());
                        return Ok(());
                    }
                }
            }
        }
    }
}
