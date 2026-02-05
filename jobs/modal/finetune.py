"""
Modal app for fine-tuning Noodle on instruction data with cloud GPUs.

Usage:
    # Prepare data locally
    uv run scripts/prepare_dolly.py

    # Upload corpus (base model should already be on volume from pretraining)
    modal volume put noodle-data corpus/dolly.txt /corpus/

    # Run fine-tuning
    modal run jobs/modal/finetune.py --corpus-file /data/corpus/dolly.txt --max-epochs 3

    # Download fine-tuned model
    modal volume get noodle-data /models/noodle-sft ./models/noodle-sft
"""

import modal

VOLUME_NAME = "noodle-data"

app = modal.App("noodle-finetune")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
    # Install Python and build essentials
    .apt_install(
        "python3",
        "python3-pip",
        "build-essential",
        "gcc",
        "pkg-config",
        "libssl-dev",
        "libvulkan1",
        "libvulkan-dev",
        "curl",
    )
    .run_commands("ln -sf /usr/bin/python3 /usr/bin/python")
    # Install Rust toolchain
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "echo 'source $HOME/.cargo/env' >> ~/.bashrc",
    )
    # Copy only what's needed for compilation
    .add_local_file("Cargo.toml", "/noodle/Cargo.toml", copy=True)
    .add_local_file("Cargo.lock", "/noodle/Cargo.lock", copy=True)
    .add_local_file("chat.rs", "/noodle/chat.rs", copy=True)
    .add_local_file("eval.rs", "/noodle/eval.rs", copy=True)
    .add_local_file("finetune.rs", "/noodle/finetune.rs", copy=True)
    .add_local_file("inference.rs", "/noodle/inference.rs", copy=True)
    .add_local_file("lib.rs", "/noodle/lib.rs", copy=True)
    .add_local_file("main.rs", "/noodle/main.rs", copy=True)
    .add_local_file("model.rs", "/noodle/model.rs", copy=True)
    .add_local_file("tokenizer.rs", "/noodle/tokenizer.rs", copy=True)
    .add_local_file("train.rs", "/noodle/train.rs", copy=True)
    # Pre-compile release binary during image build
    .run_commands(
        "cd /noodle && . $HOME/.cargo/env && cargo build --release",
        gpu="a10g",
    )
)


@app.function(
    image=image,
    gpu="a10g",
    volumes={"/data": volume},
    timeout=86400,  # 24 hours max
)
def finetune_noodle(
    corpus_file: str,
    base_model_dir: str = "/data/models/noodle",
    output_dir: str = "/data/models/noodle-sft",
    max_epochs: int = 5,
):
    """Fine-tune Noodle on instruction data using Modal GPU."""
    import subprocess
    import os

    # Find the base model file
    base_model_path = os.path.join(base_model_dir, "model.mpk")
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(
            f"Base model not found: {base_model_path}\n"
            f"Train a base model first, or check the --base-model-dir path."
        )

    # Verify corpus exists
    if not os.path.exists(corpus_file):
        raise FileNotFoundError(
            f"Corpus file not found: {corpus_file}\n"
            f"Upload with: modal volume put {VOLUME_NAME} <local-file> /corpus/"
        )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Build the fine-tuning command
    cmd = [
        "/noodle/target/release/noodle",
        "finetune",
        base_model_path,
        corpus_file,
        output_dir,
        "--backend",
        "cuda",
        "--max-epochs",
        str(max_epochs),
    ]

    print(f"Running: {' '.join(cmd)}")
    print(f"Base model: {base_model_path}")
    print(f"Corpus: {corpus_file}")
    print(f"Output dir: {output_dir}")
    print(f"Max epochs: {max_epochs}")
    print("-" * 60)

    # Run fine-tuning
    env = os.environ.copy()
    env["HOME"] = "/root"
    env["XDG_RUNTIME_DIR"] = "/tmp/runtime"
    env["RUST_BACKTRACE"] = "1"
    env["RUST_LOG"] = "wgpu=info,burn=info"
    env["WGPU_BACKEND"] = "vulkan"
    os.makedirs("/tmp/runtime", exist_ok=True)

    result = subprocess.run(
        cmd,
        env=env,
        cwd="/noodle",
    )

    # Commit volume to persist changes
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Fine-tuning failed with return code {result.returncode}")

    print("-" * 60)
    print("Fine-tuning complete!")
    print(f"Model saved to: {output_dir}")
    print(f"Download with: modal volume get {VOLUME_NAME} /models/noodle-sft ./models/noodle-sft")


@app.local_entrypoint()
def main(
    corpus_file: str,
    base_model_dir: str = "/data/models/noodle",
    output_dir: str = "/data/models/noodle-sft",
    max_epochs: int = 5,
):
    """Fine-tune Noodle on instruction data using Modal GPU.

    Args:
        corpus_file: Path to instruction corpus in the volume (e.g., /data/corpus/dolly.txt)
        base_model_dir: Directory with pre-trained base model (default: /data/models/noodle)
        output_dir: Directory to save fine-tuned model (default: /data/models/noodle-sft)
        max_epochs: Maximum fine-tuning epochs (default: 5)
    """
    finetune_noodle.remote(
        corpus_file=corpus_file,
        base_model_dir=base_model_dir,
        output_dir=output_dir,
        max_epochs=max_epochs,
    )
