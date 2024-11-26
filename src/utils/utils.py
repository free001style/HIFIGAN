from hydra.utils import instantiate


def get_losses(config, device):
    losses = instantiate(config.loss)
    for loss in losses:
        losses[loss] = losses[loss].to(device)
    return losses


def get_optimizers(config, model):
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
