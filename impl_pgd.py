import torch
import torch.nn as nn

from attack_base import Attack


class PGD(Attack):
    def __init__(self, model, config):
        super().__init__("PGD", model)
        self.eps = config["eps"]
        self.alpha = config["alpha"]
        self.steps = config["steps"]
        self.random_start = config["random_start"]
        self.targeted = config["targeted"]
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, images, labels=None, target_labels=None):
        """
        images: tensor in [0,1]
        """
        images = images.detach().to(self.device)

        if self.targeted:
            target_labels = target_labels.to(self.device)
        else:
            labels = labels.to(self.device)

        if self.random_start:
            delta = torch.empty_like(images).uniform_(-self.eps, self.eps)
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

            adv = adv.detach() + self.alpha * grad.sign()
            delta = torch.clamp(
                adv - images,
                min=-self.eps,
                max=self.eps,
            )
            adv = torch.clamp(images + delta, 0, 1)

        return adv.detach()
