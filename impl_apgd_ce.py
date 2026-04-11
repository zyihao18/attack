import torch
import torch.nn.functional as F

from attack_base import Attack


class APGD_CE_Linf(Attack):
    """
    APGD (Auto-PGD) L_inf with CE loss, closer to AutoAttack style.

    Key official-ish behaviors:
      - Nesterov/momentum update: x_{t+1} = x_t + 0.75*(x_t - x_{t-1}) +/- step*sign(grad)
      - per-sample best tracking and rollback
      - oscillation check + no-improvement check to trigger step halving
      - checkpoint schedule: k=n_iter_2; after each reduction k=max(k-size_decr, n_iter_min)
    """

    def __init__(self, model, config):
        super().__init__("APGD_CE_Linf_AA_Officialish", model)
        self.eps = float(config["eps"])
        self.steps = int(round(config["steps"]))
        self.alpha = (
            None if config.get("alpha") is None else float(config["alpha"])
        )
        self.random_start = bool(config.get("random_start", True))
        self.targeted = bool(config["targeted"])
        self.n_restarts = int(round(config.get("n_restarts", 1)))
        self.rho = float(config.get("rho", 0.75))
        self.eot_iter = int(round(config.get("eot_iter", 1)))
        self.early_stop = bool(config.get("early_stop", True))

    @torch.no_grad()
    def _success(self, logits, labels, targeted):
        pred = logits.argmax(dim=1)
        return pred.eq(labels) if targeted else ~pred.eq(labels)

    def _oscillation_check(self, loss_hist, it, k, maximize=True):
        """
        Official-ish oscillation check over last k steps:
        count how many times loss moves in the desired direction.
        If good_moves <= rho*k => oscillation/bad progress.
        loss_hist: (it+1, B)
        """
        if it < k + 1:
            return torch.zeros(
                loss_hist.size(1),
                dtype=torch.bool,
                device=loss_hist.device,
            )

        window = loss_hist[it - k : it + 1]
        diffs = window[1:] - window[:-1]

        if maximize:
            good = (diffs > 0).float().sum(dim=0)
        else:
            good = (diffs < 0).float().sum(dim=0)

        return good <= (self.rho * k)

    def forward(self, images, labels=None, target_labels=None):
        """
        images: (B,3,H,W) in [0,1]
        Untargeted: pass labels (true/original label)
        Targeted:   pass target_labels (desired target)
        """
        x0 = images.detach().to(self.device)

        if self.targeted:
            if target_labels is None:
                raise ValueError("Targeted APGD-CE requires target_labels.")
            y = target_labels.detach().to(self.device).long()
        else:
            if labels is None:
                raise ValueError("Untargeted APGD-CE requires labels.")
            y = labels.detach().to(self.device).long()

        batch_size = x0.size(0)
        n_iter = self.steps

        n_iter_2 = max(1, int(round(0.22 * n_iter)))
        n_iter_min = max(1, int(round(0.06 * n_iter)))
        size_decr = max(1, int(round(0.03 * n_iter)))

        step_size_init = (2.0 * self.eps) if (self.alpha is None) else self.alpha

        best_adv_overall = x0.clone()
        best_succ_overall = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=self.device,
        )
        if self.targeted:
            best_loss_overall = torch.full(
                (batch_size,),
                float("inf"),
                device=self.device,
            )
        else:
            best_loss_overall = torch.full(
                (batch_size,),
                float("-inf"),
                device=self.device,
            )

        for _ in range(self.n_restarts):
            if self.random_start:
                delta = torch.empty_like(x0).uniform_(-self.eps, self.eps)
                x = torch.clamp(x0 + delta, 0, 1)
            else:
                x = x0.clone()

            x_prev = x.clone()
            step_size = torch.full(
                (batch_size, 1, 1, 1),
                step_size_init,
                device=self.device,
            )

            x_best = x.clone()
            if self.targeted:
                loss_best = torch.full(
                    (batch_size,),
                    float("inf"),
                    device=self.device,
                )
            else:
                loss_best = torch.full(
                    (batch_size,),
                    float("-inf"),
                    device=self.device,
                )

            loss_best_last_check = loss_best.clone()
            loss_hist = torch.zeros((n_iter + 1, batch_size), device=self.device)

            k = n_iter_2
            last_check = 0

            for it in range(1, n_iter + 1):
                x.requires_grad_(True)

                grad_acc = torch.zeros_like(x)
                ce_acc = torch.zeros((batch_size,), device=self.device)

                for _ in range(self.eot_iter):
                    logits = self.model(x)
                    ce_vec = F.cross_entropy(logits, y, reduction="none")
                    loss = ce_vec.mean()
                    grad = torch.autograd.grad(
                        loss,
                        x,
                        retain_graph=False,
                        create_graph=False,
                    )[0]
                    grad_acc += grad.detach()
                    ce_acc += ce_vec.detach()

                grad = grad_acc / float(self.eot_iter)
                ce_vec = ce_acc / float(self.eot_iter)

                loss_hist[it] = ce_vec

                if self.early_stop:
                    with torch.no_grad():
                        if self._success(logits, y, self.targeted).all():
                            x = x.detach()
                            break

                x_mom = x.detach() + 0.75 * (x.detach() - x_prev.detach())

                if self.targeted:
                    x_new = x_mom - step_size * grad.sign()
                else:
                    x_new = x_mom + step_size * grad.sign()

                delta = torch.clamp(x_new - x0, -self.eps, self.eps)
                x_new = torch.clamp(x0 + delta, 0, 1)

                x_prev = x.detach()
                x = x_new.detach()

                with torch.no_grad():
                    logits_new = self.model(x)
                    ce_new = F.cross_entropy(logits_new, y, reduction="none")

                    if self.targeted:
                        improved = ce_new < loss_best
                    else:
                        improved = ce_new > loss_best

                    x_best[improved] = x[improved]
                    loss_best[improved] = ce_new[improved]

                if (it - last_check) >= k:
                    last_check = it

                    maximize = not self.targeted
                    osc = self._oscillation_check(loss_hist, it, k, maximize=maximize)

                    with torch.no_grad():
                        if self.targeted:
                            no_improve = loss_best >= loss_best_last_check - 1e-12
                        else:
                            no_improve = loss_best <= loss_best_last_check + 1e-12

                    reduce = osc | no_improve
                    if reduce.any():
                        step_size[reduce] = step_size[reduce] * 0.5
                        x[reduce] = x_best[reduce].clone()
                        x_prev[reduce] = x_best[reduce].clone()

                    loss_best_last_check = loss_best.clone()
                    k = max(n_iter_min, k - size_decr)

            with torch.no_grad():
                logits_best = self.model(x_best)
                succ = self._success(logits_best, y, self.targeted)
                ce_best = F.cross_entropy(logits_best, y, reduction="none")

                if self.targeted:
                    take = succ & (~best_succ_overall)
                    take |= succ & best_succ_overall & (ce_best < best_loss_overall)
                    take |= (
                        (~best_succ_overall)
                        & (~succ)
                        & (ce_best < best_loss_overall)
                    )
                else:
                    take = succ & (~best_succ_overall)
                    take |= succ & best_succ_overall & (ce_best > best_loss_overall)
                    take |= (
                        (~best_succ_overall)
                        & (~succ)
                        & (ce_best > best_loss_overall)
                    )

                best_adv_overall[take] = x_best[take]
                best_succ_overall[take] = succ[take]
                best_loss_overall[take] = ce_best[take]

        return best_adv_overall.detach()
