"""
Modal app for training Noodle on cloud GPUs.

Usage:
    # Upload training corpus
    modal volume put noodle-data corpus/tinystories-train.txt /corpus/

    # Run training
    modal run jobs/modal/train.py --max-epochs 20

    # List volume contents
    modal run jobs/modal/train.py::list_volume

    # Download trained model
    modal volume get noodle-data /models/noodle ./models/noodle
"""

import modal

VOLUME_NAME = "noodle-data"

app = modal.App("noodle-train")

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
def train_noodle(
    corpus_file: str = "/data/corpus/tinystories-train.txt",
    model_dir: str = "/data/models/noodle",
    max_epochs: int = 10,
):
    """Train Noodle on Modal GPU."""
    import subprocess
    import os

    # Verify corpus exists
    if not os.path.exists(corpus_file):
        raise FileNotFoundError(
            f"Corpus file not found: {corpus_file}\n"
            f"Upload with: modal volume put {VOLUME_NAME} <local-file> /corpus/"
        )

    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Build the training command
    cmd = [
        "/noodle/target/release/noodle",
        "train",
        corpus_file,
        model_dir,
        "--backend",
        "cuda",
        "--max-epochs",
        str(max_epochs),
    ]

    print(f"Running: {' '.join(cmd)}")
    print(f"Corpus: {corpus_file}")
    print(f"Model dir: {model_dir}")
    print(f"Max epochs: {max_epochs}")
    print("-" * 60)

    # Run training
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
        raise RuntimeError(f"Training failed with return code {result.returncode}")

    print("-" * 60)
    print("Training complete!")
    print(f"Model saved to: {model_dir}")
    print(f"Download with: modal volume get {VOLUME_NAME} /models/noodle ./models/noodle")


@app.function(image=modal.Image.debian_slim(), volumes={"/data": volume})
def list_volume(path: str = "/data"):
    """List contents of the noodle-data volume."""
    import os

    print(f"Contents of {path}:")
    print("-" * 40)

    for root, dirs, files in os.walk(path):
        level = root.replace(path, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = "  " * (level + 1)
        for file in files:
            filepath = os.path.join(root, file)
            size = os.path.getsize(filepath)
            if size > 1024 * 1024:
                size_str = f"{size / (1024 * 1024):.1f}MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f}KB"
            else:
                size_str = f"{size}B"
            print(f"{sub_indent}{file} ({size_str})")


@app.local_entrypoint()
def main(
    corpus_file: str = "/data/corpus/tinystories-train.txt",
    model_dir: str = "/data/models/noodle",
    max_epochs: int = 10,
):
    """Train Noodle on Modal GPU.

    Args:
        corpus_file: Path to corpus file in the volume (default: /data/corpus/tinystories-train.txt)
        model_dir: Directory to save model in the volume (default: /data/models/noodle)
        max_epochs: Maximum training epochs (default: 10)
    """
    train_noodle.remote(
        corpus_file=corpus_file,
        model_dir=model_dir,
        max_epochs=max_epochs,
    )
