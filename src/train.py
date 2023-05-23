import json
import logging
import os
import random
import shutil
from argparse import ArgumentParser

import gin
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from advTrainer import AdvTrainer
from config import PROJECT_ROOT
from dataset.build_dataset import build_dataset
from model.create_model import create_model
from util.optimizer_scheduler import build_optimizer, build_scheduler


@gin.configurable
def main(parameter, seed, dataset, epoch, restore_ckpt=None, aux_dataset_path=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % (parameter["cuda"])
    torch.multiprocessing.set_sharing_strategy("file_system")
    dist_training = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if dataset == "cifar10":
        n_class = 10
    elif dataset == "cifar100":
        n_class = 100
    elif dataset == "tiny-imagenet":
        n_class = 200
    else:
        raise NotImplementedError("No such dataset: %s" % (dataset))

    train_set = build_dataset(
        dataset_name=dataset,
        is_train=True,
    )
    val_set = build_dataset(
        dataset_name=dataset,
        is_train=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=parameter["batch_size"],
        num_workers=parameter["num_workers"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=parameter["batch_size"],
        num_workers=parameter["num_workers"],
        shuffle=False,
    )
    if aux_dataset_path is not None:
        aux_set = build_dataset(
            dataset_name=dataset,
            is_train=True,
            aux_dataset_path=aux_dataset_path,
        )
        aux_loader = torch.utils.data.DataLoader(
            aux_set,
            batch_size=parameter["aux_batch_size"],
            num_workers=parameter["num_workers"],
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )
    else:
        aux_loader = None

    model = create_model(dataset_name=dataset)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = build_optimizer(model, use_extra_data=aux_loader is not None)
    scheduler = build_scheduler(optimizer, epoch, len(train_loader))

    start_epoch = 0
    if restore_ckpt is not None:
        ckpt = torch.load(args.model_path)
        optimizer = ckpt["optimizer"]
        scheduler = ckpt["scheduler"]
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])

    EXP_DIR = os.path.join(PROJECT_ROOT, dataset + "_experiment")
    if not os.path.exists(EXP_DIR):
        os.mkdir(EXP_DIR)

    ckpt_path = os.path.join(EXP_DIR, parameter["description"])
    writer = None
    hparam = {
        "description": parameter["description"],
        "batch_size": parameter["batch_size"],
        "aux_batch_size": parameter["aux_batch_size"],
        "dataset": dataset,
        "n_class": n_class,
    }

    logging.info("Using device %s" % (device))
    logging.info("Dataset: %s" % (dataset))
    logging.info("Total number of params: %d" % (n_parameters))
    logging.info("create model class %d" % n_class)

    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    json.dump(
        parameter,
        open(os.path.join(ckpt_path, "config.json"), "w"),
        indent=4,
        sort_keys=True,
    )
    shutil.copy(parameter["gin_config"], os.path.join(ckpt_path, "config.gin"))
    writer = SummaryWriter(
        os.path.join(EXP_DIR, "runs/%s" % (parameter["description"]))
    )
    logging.info("Start Training")

    trainer = AdvTrainer(
        device,
        hparam,
        use_ema=parameter["ema"],
        aux_loader=aux_loader,
    )
    trainer.train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        ckpt_path,
        writer,
        epoch,
        start_epoch=start_epoch,
    )


def _parse_argument():
    parser = ArgumentParser()
    parser.add_argument(
        "--gin_config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--restore_ckpt",
        type=str,
        default=None,
        help="Path to restore from the checkpoint trained half way",
    )
    parser.add_argument("--description", type=str, required=True)
    parser.add_argument("--cuda", type=str, default="4")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--aux_batch_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--ema", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    loglevel = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=loglevel,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_argument()
    parameter = vars(args)
    parameter["description"] = args.description
    parameter["cuda"] = args.cuda
    parameter["gin_config"] = os.path.join(PROJECT_ROOT, parameter["gin_config"])
    gin.parse_config_files_and_bindings([parameter["gin_config"]], None)
    main(parameter, restore_ckpt=args.restore_ckpt)
