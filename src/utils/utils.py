from pathlib import Path

from hydra.utils import instantiate

from src.utils.io_utils import ROOT_PATH


def get_losses(config, device):
    """
    Create loss for generator and discriminators.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
    Returns:
        losses (dict[loss]): dict containing losses by key.
    """
    losses = instantiate(config.loss)
    for loss in losses:
        losses[loss] = losses[loss].to(device)
    return losses


def get_optimizers(config, model):
    """
    Create optimizers for generator and discriminators.

    Args:
        config (DictConfig): hydra experiment config.
        model (HiFiGAN): model with parameters to optimize.
    Returns:
        optimizers (dict[optimizer]): dict containing optimizers by key.
    """
    g_trainable_params = filter(lambda p: p.requires_grad, model.Generator.parameters())
    d_trainable_params = filter(
        lambda p: p.requires_grad,
        list(model.MultiScaleDiscriminator.parameters())
        + list(model.MultiPeriodDiscriminator.parameters()),
    )
    return {
        "g_optimizer": instantiate(
            config.optimizer.g_optimizer, params=g_trainable_params
        ),
        "d_optimizer": instantiate(
            config.optimizer.d_optimizer, params=d_trainable_params
        ),
    }


def get_lr_schedulers(config, optimizers):
    """
    Create lr schedulers for generator and discriminators.

    Args:
        config (DictConfig): hydra experiment config.
        optimizers (dict[optimizer]): optimizers for schedulers.
    Returns:
        optimizers (dict[lr_scheduler]): dict containing lr_scheduler by key.
    """
    return {
        "g_lr_scheduler": instantiate(
            config.lr_scheduler.g_lr_scheduler, optimizer=optimizers["g_optimizer"]
        ),
        "d_lr_scheduler": instantiate(
            config.lr_scheduler.d_lr_scheduler, optimizer=optimizers["d_optimizer"]
        ),
    }


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def get_text_from_dir(path):
    path = ROOT_PATH / path
    texts = []
    for path in (Path(path) / "transcriptions").iterdir():
        if path.suffix == ".txt":
            entry = {}
            with path.open() as f:
                entry["text"] = f.read().strip()
            entry["name"] = path.stem
            texts.append(entry)
    return texts
