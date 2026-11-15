import modal

app = modal.App("train_cifar100")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_sync("./")
    .workdir("/workspace")
    .env({"HF_HOME": "/runs"})
    .add_local_python_source("nano_cua")
    .add_local_file(
        "scripts/train_cifar100.py",
        "/workspace/scripts/train_cifar100.py"
    )
    .add_local_dir(
        "config/",
        "/workspace/config/"
    )
)

vol = modal.Volume.from_name(
    "nano-cua-volume",
    create_if_missing=True,

)


@app.function(
    image=image,
    gpu="L4",
    timeout=24 * 60 * 60,
    volumes={"/runs": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(run_dir: str, model: str):
    import runpy
    from pathlib import Path

    training = runpy.run_path("/workspace/scripts/train_cifar100.py")

    try:

        training["main"](model=model, run_dir=Path("/runs") / run_dir, handle_signals=False)
    finally:
        vol.commit()

@app.local_entrypoint()
def main(run_dir: str, model: str):
    assert model in ["resnet", "vit"]
    train.remote(run_dir, model)
