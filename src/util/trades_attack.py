import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


@gin.configurable
class TRADES:
    def __init__(self, eps, alpha, steps, distance="l_inf"):
        self.epsilon = eps / 255.0
        self.step_size = alpha / 255.0
        self.perturb_steps = steps
        self.distance = distance

    def attack(
        self,
        model,
        x_natural,
        y,
    ):
        # define KL-loss
        criterion_kl = nn.KLDivLoss(reduction="batchmean", log_target=True)
        batch_size = len(x_natural)

        # generate adversarial example
        x_adv = x_natural.clone().detach() + 0.001 * torch.rand_like(x_natural).detach()
        if self.distance == "l_inf":
            with torch.no_grad():
                natural_predict = model(x_natural)
                natural_prob = F.log_softmax(natural_predict, dim=1)
                natural_prob = natural_prob.detach()

            for _ in range(self.perturb_steps):
                x_adv.requires_grad = True
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), natural_prob)
                grad = torch.autograd.grad(
                    loss_kl, x_adv, retain_graph=False, create_graph=False
                )[0]

                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(
                    torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon
                )
                x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
        else:
            raise NotImplementedError

        return x_adv
