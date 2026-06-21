import time
import warnings

import hydra
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)

METHODS = [
    "admm100",
    "fista",
    "unrolled20",
    "modular_pre",
    "modular_post",
    "modular_prepost",
    "admm_realesrgan",
    "admm_realesrgan_ft",
]

MODEL_CONFIGS = ROOT_PATH / "src" / "configs" / "model"


def load_model(name):
    cfg = OmegaConf.load(MODEL_CONFIGS / f"{name}.yaml")
    if cfg.get("post") and "RealESRGAN" in cfg.post._target_:
        cfg.post.weights_path = None
    return instantiate(cfg)


@torch.no_grad()
def time_method(model, batch, n_warmup, n_repeat, cuda):
    for _ in range(n_warmup):
        model(**batch)
    if cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(n_repeat):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(**batch)
        if cuda:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return torch.tensor(ts)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference_admm100")
def main(config):
    set_random_seed(config.inferencer.seed)
    device = config.inferencer.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda = device.startswith("cuda")

    b = config.get("bench") or {}
    batch_size = b.get("batch_size", 1)
    n_warmup = b.get("n_warmup", 2)
    n_repeat = b.get("n_repeat", 10)

    dataloaders, _ = get_dataloaders(config, device)
    full = next(iter(dataloaders["test"]))
    batch = {
        "lensless": full["lensless"][:batch_size].to(device),
        "psf": full["psf"][:batch_size].to(device),
        "image_id": full["image_id"][:batch_size],
    }

    shape = tuple(batch["lensless"].shape)
    print(f"device={device}, batch={batch_size}, warmup={n_warmup}, repeat={n_repeat}, input={shape}\n")

    rows = []
    for name in METHODS:
        model = load_model(name).to(device).eval()
        params = sum(p.numel() for p in model.parameters()) / 1e6
        per_img = time_method(model, batch, n_warmup, n_repeat, cuda) / batch_size
        ms = per_img.mean().item() * 1e3
        rows.append(
            {
                "method": name,
                "params_M": round(params, 3),
                "ms_per_image": round(ms, 2),
                "std_ms": round(per_img.std().item() * 1e3, 2),
                "img_per_s": round(1e3 / ms, 1),
            }
        )
        print(f"{name} -> {ms:.1f} ms/image")
        del model
        if cuda:
            torch.cuda.empty_cache()

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))
    out = ROOT_PATH / "data" / "saved" / "benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
