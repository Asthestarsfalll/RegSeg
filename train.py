import argparse
import datetime
import logging
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter

from config import Config
from metrics import ConfusionMatrix
from train_utils import (
    get_data_loader,
    get_epochs_to_save,
    get_loss,
    get_lr_scheduler,
    get_model,
    get_optimizer,
)

logger = logging.getLogger()


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--tag",
        type=str,
        default=time.strftime("%Y-%m-%d-%H-%M-%S"),
    )
    parse.add_argument(
        "--config", type=str, default="configs/cityscapes_base_regseg_1000epochs.yaml"
    )
    return parse.parse_args()


def setup_env(seed=None):
    torch.backends.cudnn.benchmark = True
    seed = 0 if seed is None else seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def setup_logger(logpth, logname):
    logfile = osp.join(logpth, logname + ".log")
    if os.path.exists(logfile):
        config_name = logname.split("-")[0]
        date_time = time.strftime("%Y-%m-%d-%H-%M-%S")
        print(f"{logfile} exists, rename to {config_name}-{date_time}.log")
        logfile = osp.join(logpth, f"{config_name}-{date_time}.log")
    FORMAT = "%(levelname)s : %(message)s"
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())


def check_config(config, args):
    config_name = args.config.split(".")[0].split("/")[1]
    save_dir = osp.join(config.save_dir, config_name, args.tag)
    log_dir = osp.join(config.log_dir, config_name, args.tag)
    tensorboard_dir = osp.join(config.tensorboard_dir, config_name, args.tag)
    config.save_dir = save_dir
    config.log_dir = log_dir
    config.tensorboard_dir = tensorboard_dir
    # assert unique
    os.makedirs(save_dir, exist_ok=False)
    os.makedirs(log_dir, exist_ok=False)
    os.makedirs(tensorboard_dir, exist_ok=False)
    return config


def train_one_epoch(
    model,
    loss_fun,
    optimizer,
    loader,
    lr_scheduler,
    print_every,
    mixed_precision,
    scaler,
    epoch,
    writer,
):
    model.train()
    losses = 0
    num_iter = len(loader)
    for step, (image, target) in enumerate(loader):
        image, target = image.cuda(), target.cuda()
        with amp.autocast(enabled=mixed_precision):
            output = model(image)
            loss = loss_fun(output, target)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        losses += loss.item()
        if (step + 1) % print_every == 0:
            global_iters = step + 1 + epoch * num_iter
            writer.add_scalar("loss", loss, global_step=global_iters)
            lrs = lr_scheduler.get_last_lr()
            lrs_msg = ""
            for i in lrs:
                lrs_msg += f"{i:.5f}, "
            logger.info(
                f"[Train]: Epoch: {epoch}, Step: {step + 1}, lrs: {lrs_msg} Loss: {loss.item(): 4f}"
            )
    logger.info(f"Epoch {epoch} down, average loss: {losses / num_iter}")
    writer.add_scalar("average", losses / num_iter, global_step=epoch)


def evaluate(model, data_loader, device, confmat, mixed_precision, max_eval):
    logger.info("Begin evaluating...")
    model.eval()
    assert isinstance(confmat, ConfusionMatrix)
    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            image, target = image.to(device), target.to(device)
            with amp.autocast(enabled=mixed_precision):
                output = model(image)
            output = F.interpolate(
                output, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
            confmat.update(target.flatten(), output.argmax(1).flatten())
            if i + 1 == max_eval:
                logger.info(f"[Eval]: eval end at step {i}")
                break
    return confmat


def train(config, args, writer):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logger.info("Build dataloader...")
    train_loader, val_loader, train_set = get_data_loader(config)

    logger.info("Build model...")
    model, model_cfg = get_model(config)
    model = model.to(device)
    logger.info(model_cfg)

    logger.info("Build optimizer...")
    optim = get_optimizer(model, config)

    total_iters = len(train_loader) * config.max_epochs
    logger.info("Build lr scheduler...")
    lr_func = get_lr_scheduler(config, total_iters)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_func)

    logger.info("Build loss function...")
    loss_func = get_loss(config)

    logger.info("Build GradScaler...")
    scaler = amp.GradScaler(enabled=config.mixed_precision)

    start_time = time.time()
    best_global_accuracy = 0.0
    best_miou = 0.0
    best_reduced_miou = 0.0

    save_best_on_epochs = get_epochs_to_save(config)
    logger.info(f"model will save in epoch {save_best_on_epochs}")

    for epoch in range(config.max_epochs):
        # torch.manual_seed(epoch)
        # random.seed(epoch)
        # np.random.seed(epoch)
        if hasattr(train_set, "build_epoch"):
            logger.info("Build epoch...")
            train_set.build_epoch()
        train_one_epoch(
            model,
            loss_func,
            optim,
            train_loader,
            lr_scheduler,
            config.train_print_every,
            config.mixed_precision,
            scaler=scaler,
            epoch=epoch,
            writer=writer,
        )

        if epoch in save_best_on_epochs:
            confmat = ConfusionMatrix(config.num_classes, config.exclude_classes)
            confmat = evaluate(
                model,
                val_loader,
                device,
                confmat,
                config.mixed_precision,
                config.max_eval,
            )
            logger.info(f"[Eval]: Epoch: {epoch}, {confmat}")
            acc_global, acc, iu = confmat.compute()
            mIoU = sum(iu) / len(iu)
            reduced_iou = [
                iu[i]
                for i in range(confmat.num_classes)
                if i not in confmat.exclude_classes
            ]
            reduced_iou = sum(reduced_iou) / len(reduced_iou)
            writer.add_scalar("miou", mIoU, global_step=epoch)
            writer.add_scalar("reduced_miou", reduced_iou, global_step=epoch)
            writer.add_scalar("accuracy", acc_global, global_step=epoch)
            if mIoU > best_miou:
                best_miou = mIoU
                torch.save(
                    model.state_dict(),
                    osp.join(
                        config.save_dir,
                        f"epoch{epoch}_iou{mIoU}_acc{acc_global}_reduced{reduced_iou}.pth",
                    ),
                )
            if reduced_iou > best_reduced_miou:
                best_reduced_miou = reduced_iou
            if acc_global > best_global_accuracy:
                best_global_accuracy = acc_global

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Best mIOU: {best_miou}\n")
    logger.info(f"Best global accuracy: {best_global_accuracy}")
    logger.info(f"Training time {total_time_str}")


def main():
    args = parse_args()
    with open(args.config) as file:
        config = yaml.full_load(file)
    setup_env(config.get("RNG_seed", None))
    config = Config(config)
    config = check_config(config, args)
    writer = SummaryWriter(config.tensorboard_dir)
    setup_logger(
        config.log_dir, f"{args.config.split('.')[0].split('/')[1]}-{args.tag}"
    )
    logger.info(f"train with config: \n {config}")
    train(config, args, writer)


if __name__ == "__main__":
    main()
