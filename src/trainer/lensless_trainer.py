from src.trainer.trainer import Trainer
from src.utils.roi import normalize01


class LenslessTrainer(Trainer):
    def _log_batch(self, batch_idx, batch, mode="train"):
        self.writer.add_image("lensless", self._chw01(batch["lensless"][0]))
        self.writer.add_image("reconstructed", self._chw01(normalize01(batch["reconstructed"][0])))
        if "lensed" in batch:
            self.writer.add_image("lensed", self._chw01(batch["lensed"][0]))

    @staticmethod
    def _chw01(t):
        return t.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
