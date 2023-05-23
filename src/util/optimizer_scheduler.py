import gin
import torch
from timm.scheduler import CosineLRScheduler, MultiStepLRScheduler

from util.awp import AWP


def group_weight(model):
    group_decay = []
    group_no_decay = []
    for n, p in model.named_parameters():
        if "batchnorm" in n:
            group_no_decay.append(p)
        else:
            group_decay.append(p)
    assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=0.0)]
    return groups


@gin.configurable
def build_sgd_optimizer(model, use_extra_data, lr, momentum, weight_decay):
    return torch.optim.SGD(
        group_weight(model) if use_extra_data else model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )


@gin.configurable
def build_awp_optimizer(model, use_extra_data, lr, momentum, weight_decay, rho):
    base_optimizer = torch.optim.SGD
    return AWP(
        group_weight(model) if use_extra_data else model.parameters(),
        base_optimizer,
        rho=rho,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        adaptive=False,
    )


@gin.configurable
def build_cosine_scheduler(
    optimizer,
    epoch,
    step_per_epoch,
    lr_min,
    warmup_lr_init,
    warmup_t,
):
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=epoch * step_per_epoch,
        lr_min=lr_min,
        warmup_lr_init=warmup_lr_init,
        warmup_t=int(warmup_t * step_per_epoch),
    )
    return scheduler


@gin.configurable
def build_steplr_scheduler(
    optimizer,
    step_per_epoch,
    lr_step_milestone,
    lr_gamma,
):
    lr_step_milestone = [l * step_per_epoch for l in lr_step_milestone]
    scheduler = MultiStepLRScheduler(
        optimizer,
        decay_t=lr_step_milestone,
        decay_rate=lr_gamma,
    )
    return scheduler


@gin.configurable
def build_optimizer(model, optimizer_type, use_extra_data=False):
    if optimizer_type == "sgd":
        return build_sgd_optimizer(model, use_extra_data)
    elif optimizer_type == "awp":
        return build_awp_optimizer(model, use_extra_data)
    else:
        raise NotImplementedError


@gin.configurable
def build_scheduler(optimizer, epoch, step_per_epoch, scheduler_type):
    if scheduler_type == "cosinelr":
        return build_cosine_scheduler(optimizer, epoch, step_per_epoch)
    elif scheduler_type == "steplr":
        return build_steplr_scheduler(optimizer, step_per_epoch)
    else:
        raise NotImplementedError
