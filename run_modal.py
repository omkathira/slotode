"""Modal script to run baseline training on A100."""

import modal

app = modal.App("slotode-baseline")

# Volume with CLEVR_64 dataset
data_vol = modal.Volume.from_name("slotode-data")

# Volume for checkpoints (persists across runs / preemptions)
ckpt_vol = modal.Volume.from_name("slotode-ckpts", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "jax[cuda12]",
        "equinox",
        "optax",
        "diffrax",
        "numpy",
        "Pillow",
        "matplotlib",
        "mlflow",
        "scikit-learn",
        "scipy",
    )
    .add_local_dir(".", remote_path="/root/slotode", ignore=lambda p: not p.name.endswith(".py"))
)


def _train(num_iter: int):
    import subprocess
    import sys
    import os

    run_name = f"baseline_sa_s11_T{num_iter}"

    # Extract npz tarball if not already done
    if not os.path.exists("/data/CLEVR_64/train.npz"):
        os.makedirs("/data/CLEVR_64", exist_ok=True)
        print("Extracting clevr_npz.tar.gz...")
        subprocess.run(["tar", "-xzf", "/data/clevr_npz.tar.gz", "-C", "/data"], check=True)
        data_vol.commit()
        print("Done extracting.")

    print(f"Starting {run_name} (num_iter={num_iter})")

    # Auto-resume from latest checkpoint if preempted
    import glob
    ckpt_dir = f"/ckpts/{run_name}"
    resume_args = []
    ckpts = sorted(glob.glob(f"{ckpt_dir}/step_*.eqx"))
    if ckpts:
        latest = ckpts[-1]
        print(f"Resuming from {latest}")
        resume_args = ["--resume", latest]

    result = subprocess.run(
        [
            sys.executable, "-u", "train.py",
            "--model", "baseline",
            "--data_dir", "/data/CLEVR_64",
            "--num_slots", "11",
            "--total_steps", "500000",
            "--run_name", run_name,
            "--num_iter", str(num_iter),
            "--ckpt_every", "2000",
            "--ckpt_dir", ckpt_dir,
        ] + resume_args,
        cwd="/root/slotode",
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    ckpt_vol.commit()
    return result.returncode


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": data_vol, "/ckpts": ckpt_vol},
    timeout=86400,
)
def train_T3():
    return _train(3)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": data_vol, "/ckpts": ckpt_vol},
    timeout=86400,
)
def train_T4():
    return _train(4)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": data_vol, "/ckpts": ckpt_vol},
    timeout=86400,
)
def train_T5():
    return _train(5)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": data_vol, "/ckpts": ckpt_vol},
    timeout=86400,
)
def train_T6():
    return _train(6)


@app.local_entrypoint()
def main():
    train_T5.spawn()
    train_T6.spawn()
    print("T5 and T6 spawned. Check dashboard for logs.")
