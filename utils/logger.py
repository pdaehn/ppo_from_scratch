# logger.py
from pathlib import Path

from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir: str | Path) -> None:
        """
        Initialise a TensorBoard SummaryWriter.

        Args:
            log_dir: directory where TensorBoard will write event files.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_hyperparams(self, cfg: dict | DictConfig) -> None:
        """
        Log all hyperparameters as text, so they can be seen in the 'Text' tab.

        Args:
            cfg: configuration dictionary or DictConfig object.
        """
        for section, params in cfg.items():
            if isinstance(params, dict) or isinstance(params, DictConfig):
                for key, val in params.items():
                    tag = f"hyperparam/{section}/{key}"
                    self.writer.add_text(tag, str(val), global_step=0)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        Log a single scalar value.

        Args:
            tag: name of the scalar.
            value: value to log.
            step: global step value to record.
        """
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, scalars: dict, step: int, prefix: str = "") -> None:
        """
        Log multiple scalars with a common prefix.

        Args:
            scalars: dictionary of scalar values to log.
            step: global step value to record.
            prefix: common prefix for all scalar tags.
        """
        for key, val in scalars.items():
            self.writer.add_scalar(f"{prefix}/{key}", val, step)

    def close(self) -> None:
        """Flush and close the writer."""
        self.writer.flush()
        self.writer.close()
