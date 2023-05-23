import gin
import torch
import torch.nn as nn


@gin.configurable
class PGD:
    def __init__(self, eps, alpha, steps, random_start=True):
        self.eps = eps / 255
        self.alpha = alpha / 255
        self.steps = steps
        self.random_start = random_start

    def attack(self, model, images, labels):
        images = images.clone().detach()
        labels = labels.clone().detach()

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for step in range(self.steps):
            adv_images.requires_grad = True
            outputs = model(adv_images)
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
