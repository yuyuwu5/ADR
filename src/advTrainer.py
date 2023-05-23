import logging
import os
from itertools import cycle

import gin
import torch
import torch.nn.functional as F
import torchattacks
from timm.utils import ModelEmaV2
from torchmetrics import Accuracy, MeanMetric
from tqdm import tqdm

from adr import ADR
from util.awp import AWP
from util.bypass_bn import disable_running_stats, enable_running_stats, set_bn_momentum
from util.pgd_attack import PGD
from util.trades_attack import TRADES
from util.utils import save_ckpt


@gin.configurable
class AdvTrainer:
    def __init__(
        self,
        device,
        hparam,
        adv_train_mode,
        model_ema_decay,
        temperature_low,
        temperature_high,
        interpolation_low,
        interpolation_high,
        adv_attacker=None,
        adv_beta=6.0,
        use_ema=True,
        aux_loader=None,
    ):
        assert adv_train_mode in ["Normal", "PGD-AT", "TRADES"]
        self.device = device
        self.hparam = hparam
        self.num_classes = self.hparam["n_class"]

        self.epoch_loss = MeanMetric().to(self.device)
        self.clean_acc = Accuracy(task="multiclass", num_classes=self.num_classes).to(
            self.device
        )
        self.adv_acc = Accuracy(task="multiclass", num_classes=self.num_classes).to(
            self.device
        )

        self.adv_attacker = adv_attacker
        self.adv_beta = adv_beta  # Only for TRADES
        self.adv_train_mode = adv_train_mode

        self.use_ema = use_ema

        self.interpolation_low = interpolation_low
        self.interpolation_high = interpolation_high
        self.temperature_low = temperature_low
        self.temperature_high = temperature_high
        self.model_ema_decay = model_ema_decay

        self.aux_loader = None
        if aux_loader is not None:
            self.aux_loader = cycle(aux_loader)

        logging.info("Evaluate with EMA: %s" % (use_ema))
        logging.info("Use ADR: %s" % (temperature_high > 0))

    def train(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        ckpt_path,
        writer=None,
        epoch=50,
        start_epoch=0,
    ):
        best_eval_clean_acc = 0
        best_eval_adv_acc = 0
        model = model.to(self.device)

        self.model_ema = ModelEmaV2(model, decay=self.model_ema_decay)
        self.adr = (
            ADR(
                self.num_classes,
                epoch,
                len(train_loader),
                self.temperature_low,
                self.temperature_high,
                self.interpolation_low,
                self.interpolation_high,
            )
            if self.temperature_high > 0
            else None
        )

        try:
            for e in range(start_epoch, epoch):
                logging.info("Epoch: %s, Train Start" % (e))
                epoch_loss, train_adv_acc = self._train_one_epoch(
                    train_loader, model, optimizer, scheduler, e * len(train_loader)
                )
                lr = scheduler._get_lr(e * len(train_loader))[0]

                if self.use_ema:
                    eval_clean_acc, eval_adv_acc = self.eval(
                        self.model_ema.module, val_loader
                    )
                else:
                    eval_clean_acc, eval_adv_acc = self.eval(model, val_loader)

                logging.info("Train Loss: %s" % (epoch_loss))
                logging.info("Train Adv Acc: %s" % (train_adv_acc))
                logging.info("Eval Clean Acc: %s" % (eval_clean_acc))
                logging.info("Eval Adv Acc: %s" % (eval_adv_acc))

                if eval_clean_acc > best_eval_clean_acc:
                    best_eval_clean_acc = eval_clean_acc
                    save_ckpt(
                        os.path.join(ckpt_path, "best_clean_score.pt"),
                        model,
                        self.model_ema.module,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=e,
                    )
                if eval_adv_acc > best_eval_adv_acc:
                    best_eval_adv_acc = eval_adv_acc
                    save_ckpt(
                        os.path.join(ckpt_path, "best_adv_score.pt"),
                        model,
                        self.model_ema.module,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=e,
                    )
                if e in [50, 100, 150]:
                    save_ckpt(
                        os.path.join(ckpt_path, "%s.pt" % (e)),
                        model,
                        self.model_ema.module,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=e,
                    )

                writer.add_scalar("Score/acc", eval_clean_acc, e)
                writer.add_scalar("Score/adv_acc", eval_adv_acc, e)
                writer.add_scalar("Score/train_adv_acc", train_adv_acc, e)
                writer.add_scalar("Loss/train", epoch_loss, e)
                writer.add_scalar("lr", lr, e)
                writer.flush()

        except KeyboardInterrupt:
            logging.info("Keyboard Interrupt Received, Please Wait....")

        writer.add_hparams(
            self.hparam,
            {
                "hparam/top1_score": best_eval_clean_acc,
                "hparam/adv_score": best_eval_adv_acc,
                "hparam/adv_score_final": eval_adv_acc,
            },
            run_name="record",
        )
        save_ckpt(
            os.path.join(ckpt_path, "final_%s.pt" % (e)),
            model,
            self.model_ema.module,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=e,
        )

    def eval(self, model, val_loader):
        logging.info("Start Validatoin")
        model.eval()
        eval_attacker = torchattacks.PGD(
            model,
            eps=8.0 / 255.0,
            alpha=2.0 / 255.0,
            steps=10,
        )
        for data in tqdm(val_loader):
            img, label = data
            img = img.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                logit = model(img)
            self.clean_acc.update(logit, label)

            img_adv = eval_attacker(img, label)
            with torch.no_grad():
                logit_adv = model(img_adv)
            self.adv_acc.update(logit_adv, label)

        clean_acc = self.clean_acc.compute().item()
        adv_acc = self.adv_acc.compute().item()

        self.clean_acc.reset()
        self.adv_acc.reset()
        return clean_acc, adv_acc

    def _train_one_epoch(self, train_loader, model, optimizer, scheduler, global_step):
        cls_loss_fn = torch.nn.CrossEntropyLoss()
        kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        model.train()

        for idx, data in enumerate(tqdm(train_loader)):
            if global_step + idx == 0 and self.aux_loader is not None:
                set_bn_momentum(model, momentum=1.0)
            elif global_step + idx == 1 and self.aux_loader is not None:
                set_bn_momentum(model, momentum=0.01)

            img, label = data

            # When we want to use additoinal DDPM data.
            if self.aux_loader is not None:
                img_aux, label_aux = next(self.aux_loader)
                img = torch.vstack([img, img_aux])
                label = torch.hstack([label, label_aux])

            img = img.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            clean_label = label.clone()

            # Enabling ADR
            if self.adr is not None:
                label = self.adr(img, label, self.model_ema.module, global_step + idx)

            # Generate Adversarial Examples
            if self.adv_train_mode != "Normal":
                img_adv = self.adv_attacker.attack(model, img, label)

            # Train the model
            model.train()
            optimizer.zero_grad()

            if self.adv_train_mode == "Normal":
                if isinstance(optimizer, AWP):
                    # First Step
                    logits = model(img)
                    loss = cls_loss_fn(logits, label)
                    loss.backward()
                    optimizer.first_step(zero_grad=True)

                    # Second Step
                    disable_running_stats(model)
                    logits = model(img)
                    cls_loss_fn(logits, label).backward()
                    optimizer.second_step(zero_grad=True)
                    enable_running_stats(model)

                else:
                    logits = model(img)
                    loss = cls_loss_fn(logits, label)
                    loss.backward()
                    optimizer.step()

            elif self.adv_train_mode == "PGD-AT":
                if isinstance(optimizer, AWP):
                    # First Step
                    logits_adv = model(img_adv)
                    loss = cls_loss_fn(logits_adv, label)
                    loss.backward()
                    optimizer.first_step(zero_grad=True)

                    # Second Step
                    disable_running_stats(model)
                    logits = model(img_adv)
                    cls_loss_fn(logits, label).backward()
                    optimizer.second_step(zero_grad=True)
                    enable_running_stats(model)
                else:
                    logits_adv = model(img_adv)
                    loss = cls_loss_fn(logits_adv, label)
                    loss.backward()
                    optimizer.step()

            elif self.adv_train_mode == "TRADES":
                if isinstance(optimizer, AWP):
                    # First Step
                    logits = model(torch.cat([img, img_adv]))
                    logits_natural = logits[: img.shape[0]]
                    logits_adv = logits[img.shape[0] :]

                    loss_natural = cls_loss_fn(logits_natural, label)
                    loss_robust = kl_loss_fn(
                        F.log_softmax(logits_adv, dim=1),
                        F.log_softmax(logits_natural, dim=1),
                    )
                    loss = loss_natural + self.adv_beta * loss_robust
                    loss.backward()
                    optimizer.first_step(zero_grad=True)

                    # Second Step
                    disable_running_stats(model)
                    logits = model(torch.cat([img, img_adv]))
                    logits_natural = logits[: img.shape[0]]
                    logits_adv = logits[img.shape[0] :]

                    loss_natural = cls_loss_fn(logits_natural, label)
                    loss_robust = kl_loss_fn(
                        F.log_softmax(logits_adv, dim=1),
                        F.log_softmax(logits_natural, dim=1),
                    )
                    (loss_natural + self.adv_beta * loss_robust).backward()
                    optimizer.second_step(zero_grad=True)
                    enable_running_stats(model)

                else:
                    logits = model(torch.cat([img, img_adv]))
                    logits_natural = logits[: img.shape[0]]
                    logits_adv = logits[img.shape[0] :]

                    loss_natural = cls_loss_fn(logits_natural, label)
                    loss_robust = kl_loss_fn(
                        F.log_softmax(logits_adv, dim=1),
                        F.log_softmax(logits_natural, dim=1),
                    )
                    loss = loss_natural + self.adv_beta * loss_robust

                    loss.backward()
                    optimizer.step()

            scheduler.step(global_step + idx)

            if self.adv_train_mode != "Normal":
                self.adv_acc.update(logits_adv, clean_label)
            self.model_ema.update(model)
            self.epoch_loss.update(loss)

        epoch_loss = self.epoch_loss.compute().item()
        epoch_robust_acc = (
            0 if self.adv_train_mode == "Normal" else self.adv_acc.compute().item()
        )

        self.epoch_loss.reset()
        self.adv_acc.reset()
        return epoch_loss, epoch_robust_acc
