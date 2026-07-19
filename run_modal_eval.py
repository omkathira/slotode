"""Modal script to eval baseline checkpoints inside the same image used for training."""

import modal

app = modal.App("slotode-baseline-eval")

data_vol = modal.Volume.from_name("slotode-data")
ckpt_vol = modal.Volume.from_name("slotode-ckpts")

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


def _ensure_data():
    import os
    import subprocess

    if not os.path.exists("/data/CLEVR_64/val.npz"):
        os.makedirs("/data/CLEVR_64", exist_ok=True)
        print("Extracting clevr_npz.tar.gz...")
        subprocess.run(["tar", "-xzf", "/data/clevr_npz.tar.gz", "-C", "/data"], check=True)
        print("Done extracting.")


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": data_vol, "/ckpts": ckpt_vol},
    timeout=3600,
)
def eval_ckpt(run_name: str, ckpt_file: str = "best.eqx", num_samples: int = 5000):
    import subprocess
    import sys

    _ensure_data()

    ckpt_path = f"/ckpts/{run_name}/{ckpt_file}"

    print(f"\n{'='*60}")
    print(f"Evaluating {run_name} :: {ckpt_file}")
    print(f"{'='*60}\n")

    import jax, equinox
    print(f"jax={jax.__version__}  equinox={equinox.__version__}")

    result = subprocess.run(
        [
            sys.executable, "-u", "evaluate.py",
            "--ckpt", ckpt_path,
            "--model", "baseline",
            "--data_root", "/data/CLEVR_64",
            "--split", "val",
            "--num_samples", str(num_samples),
            "--batch_size", "64",
        ],
        cwd="/root/slotode",
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return result.returncode


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": data_vol, "/ckpts": ckpt_vol},
    timeout=3600,
)
def convergence_curve(run_name: str, model: str = "baseline",
                      step_multiple: int = 10000, num_samples: int = 200):
    """Post-hoc convergence curve: eval every step_multiple checkpoint on train+val."""
    import subprocess
    import sys

    _ensure_data()

    run_dir = f"/ckpts/{run_name}"
    out_path = f"/ckpts/{run_name}/convergence_curve.json"

    print(f"\nConvergence curve for {run_name} (model={model})\n")

    result = subprocess.run(
        [
            sys.executable, "-u", "convergence_curve.py",
            "--run_dir", run_dir,
            "--model", model,
            "--data_root", "/data/CLEVR_64",
            "--step_multiple", str(step_multiple),
            "--num_samples", str(num_samples),
            "--out", out_path,
        ],
        cwd="/root/slotode",
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    ckpt_vol.commit()
    print(f"Wrote {out_path} (committed to volume)")
    return result.returncode


@app.local_entrypoint()
def main():
    eval_ckpt.remote("baseline_sa_s11_T3", "best.eqx", 5000)
