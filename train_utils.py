import torch

from datasets.data import get_cityscapes
from losses import LOSSES
from lr_schedulers import LR_SCHEDULERS
from models import MODEL_CONFIGS, RegSeg

OPTIMS = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}


def get_data_loader(config):
    dataset_type = config.dataset_type
    if dataset_type == "cityscapes":
        train_loader, val_loader, train_dataset = get_cityscapes(**config.dataset_cfg)
    else:
        raise ValueError()
    return train_loader, val_loader, train_dataset


def get_model(config):
    model_config = MODEL_CONFIGS[config.model_type](config.num_classes)
    model = RegSeg(model_config)
    return model, model_config


def get_loss(config):
    loss_func = LOSSES[config.loss_type](**config.loss_cfg)
    return loss_func


def get_optimizer(model, config):
    if not config.bn_weight_decay:
        p_bn = [p for n, p in model.named_parameters() if "bn" in n]
        p_non_bn = [p for n, p in model.named_parameters() if "bn" not in n]
        optim_params = [
            {"params": p_bn, "weight_decay": 0},
            {"params": p_non_bn, "weight_decay": config.optim_cfg["weight_decay"]},
        ]
    else:
        optim_params = model.parameters()

    return OPTIMS[config.optim_type](optim_params, **config.optim_cfg)


def get_lr_scheduler(config, total_iters):
    return lambda x: LR_SCHEDULERS[config.lr_scheduler](
        x, total_iters, **config.lr_scheduler_cfg
    )


def get_epochs_to_save(config):
    if not config.eval_while_train:
        print("warning: no checkpoint/eval during training")
        return []
    epochs = config.max_epochs
    save_every_k_epochs = config.save_every_k_epochs
    save_best_on_epochs = [
        i * save_every_k_epochs - 1 for i in range(1, epochs // save_every_k_epochs + 1)
    ]
    if epochs - 1 not in save_best_on_epochs:
        save_best_on_epochs.append(epochs - 1)
    if 0 not in save_best_on_epochs:
        save_best_on_epochs.append(0)
    if config.save_last_k_epochs:
        for i in range(max(epochs - config.save_last_k_epochs, 0), epochs):
            if i not in save_best_on_epochs:
                save_best_on_epochs.append(i)
    save_best_on_epochs = sorted(save_best_on_epochs)
    return save_best_on_epochs
