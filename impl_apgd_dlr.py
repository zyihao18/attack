import torch

from attack_base import Attack


class APGD_DLR_Linf(Attack):
    """
    Closer-to-AutoAttack APGD (L_inf) with DLR loss.

    Key AA-like features included:
      - DLR loss (untargeted) and a practical targeted DLR-style loss
      - per-sample best tracking (x_best, loss_best)
      - oscillation check in a sliding window
      - step-size halving + rollback to best when oscillation/no-improve
      - checkpoint schedule similar to AutoAttack (n_iter_2, n_iter_min)

    Notes:
      - AutoAttack targeted variant (apgd-t) cycles multiple targets; here we support single target label
        (modern equivalent, minimal but strong and fair for your framework).
    """

    def __init__(self, model, config):
        super().__init__("APGD_DLR_Linf_AA", model)
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

    def _dlr_untargeted(self, logits, y):
        """
        Untargeted DLR (as in AutoAttack family):
          if y is top1:
              -(z_y - z_2)/(z_1 - z_3)
          else:
              -(z_y - z_1)/(z_1 - z_3)
        We maximize this loss for untargeted attack.
        Returns (B,)
        """
        z_sorted, ind_sorted = logits.sort(dim=1, descending=True)
        z1 = z_sorted[:, 0]
        z2 = z_sorted[:, 1]
        z3 = z_sorted[:, 2]

        y = y.long()
        zy = logits.gather(1, y.view(-1, 1)).squeeze(1)

        is_top1 = (ind_sorted[:, 0] == y).float()
        num = is_top1 * (zy - z2) + (1.0 - is_top1) * (zy - z1)
        den = (z1 - z3).clamp_min(1e-12)
        return -num / den

    def _dlr_targeted(self, logits, y, y_target):
        """
        Targeted DLR close to the official AutoAttack definition:
          -(z_y - z_t) / (z_1 - 0.5 * (z_3 + z_4))
        where z_1 >= z_2 >= ...

        We maximize this loss for targeted attack, matching the official
        AutoAttack convention.
        Returns (B,)
        """
        x_sorted, _ = logits.sort(dim=1)
        y = y.long()
        y_target = y_target.long()
        u = torch.arange(logits.shape[0], device=logits.device)

        if logits.size(1) >= 4:
            den = x_sorted[:, -1] - 0.5 * (x_sorted[:, -3] + x_sorted[:, -4])
        elif logits.size(1) == 3:
            den = x_sorted[:, -1] - x_sorted[:, -3]
        elif logits.size(1) == 2:
            den = x_sorted[:, -1] - x_sorted[:, -2]
        else:
            raise ValueError("Targeted DLR requires at least 2 classes.")

        return -(logits[u, y] - logits[u, y_target]) / den.clamp_min(1e-12)

    @torch.no_grad()
    def _success(self, logits, labels, targeted):
        pred = logits.argmax(dim=1)
        return pred.eq(labels) if targeted else ~pred.eq(labels)

    def _oscillation(self, loss_hist, it, k, maximize=True):
        """
        AutoAttack-like oscillation check:
        in last k steps, count how often loss increased (for maximize) or decreased (for minimize).
        If "good moves" <= rho * k  => oscillating / not progressing well.

        loss_hist: Tensor (it+1, B) storing per-iter loss values
        it: current iteration index (1..steps)
        k: window size
        maximize: whether we want the loss to increase (untargeted) or decrease (targeted)
        Returns: bool mask (B,) where True means "oscillation / bad progress"
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
        Untargeted: call with labels
        Targeted:   call with target_labels
        """
        x0 = images.detach().to(self.device)

        if self.targeted:
            if target_labels is None:
                raise ValueError("Targeted APGD requires target_labels.")

            y_target = target_labels.detach().to(self.device).long()
            if labels is None:
                with torch.no_grad():
                    y = self.model(x0).argmax(dim=1)
            else:
                y = labels.detach().to(self.device).long()
        else:
            if labels is None:
                raise ValueError("Untargeted APGD requires labels.")
            y = labels.detach().to(self.device).long()

        batch_size = x0.size(0)
        n_iter = self.steps

        n_iter_2 = max(1, int(0.22 * n_iter))
        n_iter_min = max(1, int(0.06 * n_iter))

        if self.alpha is None:
            step_size_init = 2.0 * self.eps
        else:
            step_size_init = float(self.alpha)

        best_adv_overall = x0.clone()
        best_succ_overall = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=self.device,
        )

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

            step_size = torch.full(
                (batch_size, 1, 1, 1),
                step_size_init,
                device=self.device,
            )

            x_best = x.clone()
            loss_best = torch.full(
                (batch_size,),
                float("-inf"),
                device=self.device,
            )

            loss_hist = torch.zeros((n_iter + 1, batch_size), device=self.device)

            k = n_iter_2
            last_check = 0

            for it in range(1, n_iter + 1):
                x.requires_grad_(True)

                grad_acc = torch.zeros_like(x)
                loss_vec_acc = torch.zeros((batch_size,), device=self.device)

                for _ in range(self.eot_iter):
                    logits = self.model(x)

                    if self.targeted:
                        loss_vec = self._dlr_targeted(logits, y, y_target)
                        loss = loss_vec.mean()
                    else:
                        loss_vec = self._dlr_untargeted(logits, y)
                        loss = loss_vec.mean()

                    grad = torch.autograd.grad(
                        loss,
                        x,
                        retain_graph=False,
                        create_graph=False,
                    )[0]
                    grad_acc += grad.detach()
                    loss_vec_acc += loss_vec.detach()

                grad = grad_acc / float(self.eot_iter)
                loss_vec = loss_vec_acc / float(self.eot_iter)

                loss_hist[it] = loss_vec

                if self.early_stop:
                    with torch.no_grad():
                        succ_labels = y_target if self.targeted else y
                        succ_now = self._success(logits, succ_labels, self.targeted)
                        if succ_now.all():
                            x = x.detach()
                            break

                x_new = x.detach() + step_size * grad.sign()

                delta = torch.clamp(x_new - x0, -self.eps, self.eps)
                x_new = torch.clamp(x0 + delta, 0, 1)

                x = x_new.detach()

                with torch.no_grad():
                    logits_new = self.model(x)
                    if self.targeted:
                        lv = self._dlr_targeted(logits_new, y, y_target)
                        improved = lv > loss_best
                    else:
                        lv = self._dlr_untargeted(logits_new, y)
                        improved = lv > loss_best

                    x_best[improved] = x[improved]
                    loss_best[improved] = lv[improved]

                if (it - last_check) >= k:
                    last_check = it

                    maximize = True
                    osc = self._oscillation(loss_hist, it, k, maximize=maximize)

                    with torch.no_grad():
                        if self.targeted:
                            best_recent = loss_hist[it - k : it + 1].max(dim=0).values
                            no_improve = best_recent <= loss_best + 1e-12
                        else:
                            best_recent = loss_hist[it - k : it + 1].max(dim=0).values
                            no_improve = best_recent <= loss_best + 1e-12

                    reduce = osc | no_improve

                    if reduce.any():
                        step_size[reduce] = step_size[reduce] * 0.5
                        x[reduce] = x_best[reduce].clone()

                    k = max(n_iter_min, int(0.9 * k))

            with torch.no_grad():
                logits_best = self.model(x_best)
                succ_labels = y_target if self.targeted else y
                succ = self._success(logits_best, succ_labels, self.targeted)

                if self.targeted:
                    lv = self._dlr_targeted(logits_best, y, y_target)
                    take = succ & (~best_succ_overall)
                    take |= succ & best_succ_overall & (lv > best_loss_overall)
                    take |= (
                        (~best_succ_overall)
                        & (~succ)
                        & (lv > best_loss_overall)
                    )
                else:
                    lv = self._dlr_untargeted(logits_best, y)
                    take = succ & (~best_succ_overall)
                    take |= succ & best_succ_overall & (lv > best_loss_overall)
                    take |= (
                        (~best_succ_overall)
                        & (~succ)
                        & (lv > best_loss_overall)
                    )

                best_adv_overall[take] = x_best[take]
                best_succ_overall[take] = succ[take]
                best_loss_overall[take] = lv[take]

        return best_adv_overall.detach()
