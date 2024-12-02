from pathlib import Path

import pandas as pd
import torch
from torch import autocast

import wandb
from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["train"]
        self.optimizer["g_optimizer"].zero_grad()
        self.optimizer["d_optimizer"].zero_grad()

        with autocast(
            device_type=self.device, enabled=self.is_amp, dtype=torch.float16
        ):
            outputs = self.model.Generator(**batch)
            batch.update(outputs)
            mpd_outputs = self.model.MultiPeriodDiscriminator(
                predict=batch["predict"].detach(), audio=batch["audio"]
            )
            batch.update(mpd_outputs)
            msd_outputs = self.model.MultiScaleDiscriminator(
                predict=batch["predict"].detach(), audio=batch["audio"]
            )
            batch.update(msd_outputs)

            d_losses = self.criterion["d_loss"](**batch)
            batch.update(d_losses)

        self.scaler.scale(batch["d_loss"]).backward()
        self._clip_grad_norm("d")
        self.scaler.step(self.optimizer["d_optimizer"])
        self.scaler.update()
        metrics.update("d_grad_norm", self._get_grad_norm("d"))

        batch["spectrogram_predict"] = self.batch_transforms.get("train")[
            "spectrogram"
        ](batch["predict"])

        with autocast(
            device_type=self.device, enabled=self.is_amp, dtype=torch.float16
        ):
            mpd_outputs = self.model.MultiPeriodDiscriminator(**batch)
            batch.update(mpd_outputs)
            msd_outputs = self.model.MultiScaleDiscriminator(**batch)
            batch.update(msd_outputs)

            g_losses = self.criterion["g_loss"](**batch)
            batch.update(g_losses)

        self.scaler.scale(batch["g_loss"]).backward()
        self._clip_grad_norm("g")
        self.scaler.step(self.optimizer["g_optimizer"])
        self.scaler.update()
        metrics.update("g_grad_norm", self._get_grad_norm("g"))

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        self.log_spectrogram(**batch)
        self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, spectrogram_predict, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

        spectrogram_for_plot = spectrogram_predict[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram_predict", image)

    def log_predictions(self, audio, predict, audio_path, **batch):
        rows = {}
        for i in range(min(10, audio.shape[0])):
            rows[Path(audio_path[i]).name] = {
                "audio": wandb.Audio(
                    audio[i].detach().cpu().numpy(), sample_rate=22050
                ),
                "predict": wandb.Audio(
                    predict[i].to(torch.float32).detach().cpu().numpy(),
                    sample_rate=22050,
                ),
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
