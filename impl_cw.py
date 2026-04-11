import torch

from attack_base import Attack


class CWLinf(Attack):
    """
    CW-style margin loss (on logits) under L_infty constraint.
    - Untargeted: maximize f(x) = max_{i!=y} z_i - z_y (want > 0)
    - Targeted:   maximize f(x) = z_t - max_{i!=t} z_i (want > 0)
    Then we do gradient ascent on f(x) (or descent on -f).
    """

    def __init__(self, model, config):
        super().__init__("CW_Linf", model)
        self.eps = config["eps"]
        self.alpha = config["alpha"]
        self.steps = config["steps"]
        self.random_start = config.get("random_start", False)
        self.targeted = config["targeted"]
        self.kappa = config.get("kappa", 0.0)

    def _cw_margin(self, logits, labels, targeted):
        """
        Returns per-sample margin f(x) (B,).
        logits: (B,C)
        labels: (B,)
        """
        labels = labels.long()

        z_label = logits.gather(1, labels.view(-1, 1)).squeeze(1)

        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(1, labels.view(-1, 1), False)
        other_max = logits.masked_fill(~mask, float("-inf")).max(dim=1).values

        if targeted:
            margin = z_label - other_max
        else:
            margin = other_max - z_label

        return margin

    @torch.no_grad()
    def _success(self, logits, labels, targeted):
        pred = logits.argmax(dim=1)
        if targeted:
            return pred.eq(labels)
        return ~pred.eq(labels)

    def forward(self, images, labels=None, target_labels=None):
        """
        images: (B,3,H,W) in [0,1]
        - untargeted: pass labels
        - targeted: pass target_labels
        """
        images = images.detach().to(self.device)

        if self.targeted:
            if target_labels is None:
                raise ValueError("Targeted CWLinf requires target_labels.")
            labels_use = target_labels.detach().to(self.device)
        else:
            if labels is None:
                raise ValueError("Untargeted CWLinf requires labels.")
            labels_use = labels.detach().to(self.device)

        if self.random_start:
            delta = torch.empty_like(images).uniform_(-self.eps, self.eps)
        else:
            delta = torch.zeros_like(images)

        delta = delta.to(self.device)
        delta = delta.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([delta], lr=self.alpha)

        adv = torch.clamp(images + delta, 0, 1)

        for _ in range(self.steps):
            adv = torch.clamp(images + delta, 0, 1)
            logits = self.model(adv)

            margin = self._cw_margin(logits, labels_use, targeted=self.targeted)
            obj = margin - self.kappa
            loss = (-obj).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                delta.data.clamp_(-self.eps, self.eps)
                adv = torch.clamp(images + delta.data, 0, 1)
                delta.data = adv - images

                if self._success(logits, labels_use, targeted=self.targeted).all():
                    break

        return torch.clamp(images + delta.detach(), 0, 1).detach()
