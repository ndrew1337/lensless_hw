import numpy as np
from PIL import Image

from src.trainer.inferencer import Inferencer
from src.utils.roi import normalize01


class LenslessInferencer(Inferencer):
    def process_batch(self, batch_idx, batch, metrics, part):
        batch = self.move_batch_to_device(batch)
        if "lensed" in batch:
            batch["lensed"] = batch["lensed"].to(self.device)
        batch.update(self.model(**batch))

        if metrics is not None and "lensed" in batch:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        out_dir = self.save_path / part
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, image_id in enumerate(batch["image_id"]):
            img = normalize01(batch["reconstructed"][i]).detach().cpu()
            arr = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(arr).save(out_dir / f"{image_id}.png")
        return batch
