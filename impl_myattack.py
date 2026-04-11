import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcam.methods import ScoreCAM

from attack_base import Attack


class MyAttack(Attack):
    def __init__(self, model, config):
        super().__init__("ScoreCAM_PGD", model)

        self.eps = config["eps"]
        self.alpha = config["alpha"]
        self.steps = config["steps"]
        self.random_start = config["random_start"]
        self.targeted = config["targeted"]
        self.keep_ratio = config["cam_keep_ratio"]

        self.loss_fn = nn.CrossEntropyLoss()
        self.cam = ScoreCAM(self.model)

    @torch.no_grad()
    def get_scorecam_mask(self, images, labels):
        _ = self.model(images)

        if labels.numel() == 1:
            class_idx = int(labels.item())
        else:
            class_idx = labels.tolist()

        cam = self.cam(class_idx=class_idx)[0]

        batch_size, height, width = cam.shape
        cam_flat = cam.view(batch_size, -1)

        cam_min = cam_flat.min(dim=1, keepdim=True)[0]
        cam_max = cam_flat.max(dim=1, keepdim=True)[0]
        cam_norm = (cam_flat - cam_min) / (cam_max - cam_min + 1e-8)
        cam_norm = cam_norm.view(batch_size, height, width)

        mask = torch.zeros_like(cam_norm)

        for batch_idx in range(batch_size):
            cam_b = cam_norm[batch_idx]
            cam_levels = torch.unique(cam_b)
            cam_levels_sorted = torch.sort(cam_levels, descending=True)[0]

            num_levels = cam_levels_sorted.numel()
            k = max(1, int(self.keep_ratio * num_levels))
            selected_levels = cam_levels_sorted[:k]

            mask[batch_idx] = torch.isin(cam_b, selected_levels)

        mask = mask.float().unsqueeze(1)

        mask = F.interpolate(
            mask,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return (mask >= 0.5).float()

    def forward(self, images, labels=None, target_labels=None):
        """
        images: tensor in [0,1]
        """
        images = images.detach().to(self.device)

        if self.targeted:
            if target_labels is None:
                raise ValueError("Targeted PGD requires target_labels.")
            target_labels = target_labels.to(self.device)
            cam_mask = self.get_scorecam_mask(images, target_labels)
        else:
            labels = labels.to(self.device)
            cam_mask = self.get_scorecam_mask(images, labels)

        if self.random_start:
            delta = torch.empty_like(images).uniform_(-self.eps, self.eps)
            delta = delta * cam_mask
            adv = torch.clamp(images + delta, 0, 1)
        else:
            adv = images.clone()

        for _ in range(self.steps):
            adv.requires_grad_(True)

            outputs = self.model(adv)

            with torch.no_grad():
                pred = outputs.argmax(dim=1).item()
                if self.targeted:
                    if pred == int(target_labels.item()):
                        break
                else:
                    if pred != int(labels.item()):
                        break

            if self.targeted:
                loss = -self.loss_fn(outputs, target_labels)
            else:
                loss = self.loss_fn(outputs, labels)

            grad = torch.autograd.grad(
                loss,
                adv,
                retain_graph=False,
                create_graph=False,
            )[0]

            grad = grad * cam_mask

            adv = adv.detach() + self.alpha * grad.sign()

            delta = torch.clamp(
                adv - images,
                min=-self.eps,
                max=self.eps,
            )
            delta = delta * cam_mask
            adv = torch.clamp(images + delta, 0, 1)

        return adv.detach()
