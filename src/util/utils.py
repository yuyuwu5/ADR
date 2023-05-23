import numpy as np
import torch


def save_ckpt(save_name, model, model_ema, optimizer=None, scheduler=None, epoch=None):
    model_weight = model.state_dict()
    model_ema_weight = model_ema.state_dict()
    torch.save(
        {
            "model": model_weight,
            "model_ema": model_ema_weight,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": epoch,
        },
        save_name,
    )


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
