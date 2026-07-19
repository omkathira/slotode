"""Print versions inside modal's cached slotode-baseline image."""

import modal

app = modal.App("slotode-versions")

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
)


@app.function(image=image)
def versions():
    import subprocess
    import sys

    print("=== key versions ===")
    import jax, equinox, optax, diffrax, numpy
    print(f"jax={jax.__version__}")
    print(f"equinox={equinox.__version__}")
    print(f"optax={optax.__version__}")
    print(f"diffrax={diffrax.__version__}")
    print(f"numpy={numpy.__version__}")

    print("\n=== full pip freeze ===")
    subprocess.run([sys.executable, "-m", "pip", "freeze"])


@app.local_entrypoint()
def main():
    versions.remote()
