import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torchattacks
import torchvision
from torchmetrics import Accuracy
from tqdm import tqdm

from dataset.build_dataset import build_dataset
from model.create_model import create_model


def attack(device, model, attacker, parameter, n_class):
    val_set = build_dataset(
        dataset_name=parameter["dataset"],
        is_train=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=parameter["batch_size"],
        num_workers=4,
        drop_last=False,
        shuffle=False,
    )
    logging.info("Attack Start!")
    model.eval()
    clean_acc_metric = Accuracy(task="multiclass", num_classes=n_class).to(device)
    adv_acc_metric = Accuracy(task="multiclass", num_classes=n_class).to(device)
    for data in tqdm(val_loader):
        img, label = data
        img = img.to(device)
        label = label.to(device)
        with torch.no_grad():
            logit = model(img)
        clean_acc_metric.update(logit, label)

        adv_img = attacker(img, label)
        with torch.no_grad():
            adv_logit = model(adv_img)
        adv_acc_metric.update(adv_logit, label)
    clean_acc = clean_acc_metric.compute().item()
    adv_acc = adv_acc_metric.compute().item()
    return clean_acc, adv_acc


def main(parameter):
    logging.info(
        "Attack Model: %s from %s" % (parameter["model_type"], parameter["model_path"])
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % (parameter["cuda"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = parameter["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if parameter["dataset"] == "cifar10":
        n_class = 10
    elif parameter["dataset"] == "cifar100":
        n_class = 100
    elif parameter["dataset"] == "tiny-imagenet":
        n_class = 200
    else:
        raise NotImplementedError("No such dataset: %s" % (parameter["dataset"]))

    model = create_model(
        parameter["model_type"],
        dataset_name=parameter["dataset"],
        activation_name=parameter["activation_name"],
    )
    if parameter["model_path"] is not None:
        logging.info("Loading weight from checkpoint.....")
        key = "model_ema" if parameter["ema"] else "model"
        model.load_state_dict(torch.load(args.model_path, map_location=device)[key])
    else:
        raise ValueError("No such checkpoint")

    model.to(device)
    model.eval()
    if parameter["attack_type"] == "fgsm":
        attacker = torchattacks.FGSM(model, eps=parameter["epsilon"] / 255)
        logging.info(
            "Attack type: %s, epsilon: %.5f"
            % (
                parameter["attack_type"],
                parameter["epsilon"] / 255,
            )
        )

    elif parameter["attack_type"] == "pgd":
        attacker = torchattacks.PGD(
            model,
            eps=parameter["epsilon"] / 255,
            alpha=parameter["alpha"] / 255,
            steps=parameter["steps"],
        )
        logging.info(
            "Attack type: %s, epsilon: %.5f, alpha: %.5f, step: %d"
            % (
                parameter["attack_type"],
                parameter["epsilon"] / 255,
                parameter["alpha"] / 255,
                parameter["steps"],
            )
        )
    elif parameter["attack_type"] == "autoattack":
        attacker = torchattacks.AutoAttack(
            model,
            eps=parameter["epsilon"] / 255,
            n_classes=n_class,
            seed=parameter["seed"],
        )
        logging.info(
            "Attack type: %s, epsilon: %.5f"
            % (
                parameter["attack_type"],
                parameter["epsilon"] / 255,
            )
        )
    elif parameter["attack_type"] == "square":
        logging.info(
            "Attack type: %s, epsilon: %.5f"
            % (
                parameter["attack_type"],
                parameter["epsilon"] / 255,
            )
        )
        attacker = torchattacks.Square(
            model,
            eps=parameter["epsilon"] / 255.0,
            n_queries=5000,
            n_restarts=1,
            seed=parameter["seed"],
        )
    else:
        raise NotImplementedError

    clean_acc, adv_acc = attack(device, model, attacker, parameter, n_class)
    logging.info("Clean accuracy: %.4f" % (clean_acc))
    logging.info("Adv accuracy: %.4f" % (adv_acc))


def _parse_argument():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "tiny-imagenet"],
    )
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--activation_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--attack_type", type=str, default="fgsm")
    parser.add_argument("--epsilon", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=str, default="4")
    parser.add_argument("--batch_size", type=int, default=64)
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
    main(parameter)
