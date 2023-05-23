import torch
import torch.nn.functional as F

from util.utils import cosine_scheduler


class ADR:
    def __init__(
        self,
        num_classes,
        total_epoch,
        train_loader_length,
        temperature_low,
        temperature_high,
        interpolation_low,
        interpolation_high,
    ):
        self.num_classes = num_classes
        self.temperature_scheduler = cosine_scheduler(
            temperature_high,
            temperature_low,
            total_epoch,
            train_loader_length,
        )
        self.interpolation_scheduler = cosine_scheduler(
            interpolation_low,
            interpolation_high,
            total_epoch,
            train_loader_length,
        )

    def __call__(self, img, label, teacher_model, step):
        with torch.no_grad():
            one_hot_label = torch.nn.functional.one_hot(
                label, num_classes=self.num_classes
            )
            logits_teacher = teacher_model(img)
            logits_teacher /= self.temperature_scheduler[step]
            prob_teacher = F.softmax(logits_teacher, dim=1)

            target_prob = torch.gather(prob_teacher, 1, label.unsqueeze(1)).squeeze()
            max_prob, _ = torch.max(prob_teacher, 1)
            prob_per_img = self.interpolation_scheduler[step] - (max_prob - target_prob)
            prob_per_img = torch.clamp(prob_per_img, min=0.0, max=1.0)
            prob_per_img = prob_per_img.unsqueeze(1)
            rectified_label = (
                prob_per_img * prob_teacher + (1 - prob_per_img) * one_hot_label
            )
        return rectified_label
