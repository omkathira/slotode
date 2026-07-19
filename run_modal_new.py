"""Modal script to train SlotODE with the simplified MLP (model_new.py)."""

import modal

app = modal.App("slotode-new")

data_vol = modal.Volume.from_name("slotode-data")
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


def _train(num_iter: int, dt: float):
    import subprocess
    import sys
    import os
    import glob

    run_name = f"slotode_new_T{num_iter}_dt{dt}"

    if not os.path.exists("/data/CLEVR_64/train.npz"):
        os.makedirs("/data/CLEVR_64", exist_ok=True)
        print("Extracting clevr_npz.tar.gz...")
        subprocess.run(["tar", "-xzf", "/data/clevr_npz.tar.gz", "-C", "/data"], check=True)
        data_vol.commit()
        print("Done extracting.")

    print(f"Starting {run_name} (num_iter={num_iter}, dt={dt})")

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
            "--model", "slot_ode_new",
            "--data_dir", "/data/CLEVR_64",
            "--num_slots", "11",
            "--total_steps", "500000",
            "--run_name", run_name,
            "--num_iter", str(num_iter),
            "--dt", str(dt),
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
def train_T3_dt1():
    return _train(3, 1.0)


@app.local_entrypoint()
def main():
    train_T3_dt1.spawn()
    print("slotode_new T=3 dt=1 spawned. Check dashboard for logs.")
