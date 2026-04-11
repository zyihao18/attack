import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def count_modified_pixels(adv, clean, threshold=0.0):
    """
    adv, clean: (1, 3, H, W), value range [0,1]
    """
    delta = (adv - clean).abs()
    delta_hw = delta.max(dim=1)[0]

    modified_mask = delta_hw > threshold
    modified_pixels = modified_mask.sum().item()
    total_pixels = delta_hw.numel()

    return modified_pixels, total_pixels


@torch.no_grad()
def is_attack_success(model, adv_images, labels, targeted):
    outputs = model(adv_images)
    preds = outputs.argmax(dim=1)

    if targeted:
        return preds.eq(labels).item()

    return not preds.eq(labels).item()


def cumulative_gradient_pruning(
    model,
    adv_images,
    clean_images,
    labels,
    targeted=False,
    step_ratio=0.01,
    max_ratio=1.0,
):
    """
    返回：
        adv_pruned : 剪枝后的对抗样本（[0,1]）
        final_mask : 最终保留的扰动 mask (B,1,H,W)
    """

    adv = adv_images.detach().clone().requires_grad_(True)
    clean = clean_images.detach()
    delta = adv - clean

    outputs = model(adv)

    if targeted:
        loss = -F.cross_entropy(outputs, labels)
    else:
        loss = F.cross_entropy(outputs, labels)

    grad = torch.autograd.grad(loss, adv)[0]

    score = (delta * grad).abs().sum(dim=1)

    batch_size, height, width = score.shape
    total_pixels = height * width

    score_flat = score.view(batch_size, -1)
    sorted_idx = torch.argsort(score_flat, dim=1, descending=True)

    step = max(1, int(step_ratio * total_pixels))
    max_steps = int(max_ratio * total_pixels)

    current_mask = torch.zeros_like(score_flat)

    for k in range(step, max_steps + step, step):
        idx = sorted_idx[:, :k]

        current_mask.zero_()
        current_mask.scatter_(1, idx, 1.0)

        mask = current_mask.view(batch_size, 1, height, width)

        delta_k = delta * mask
        adv_k = torch.clamp(clean + delta_k, 0, 1)

        if is_attack_success(model, adv_k, labels, targeted):
            return adv_k.detach(), mask.detach()

    full_mask = torch.ones_like(score_flat).view(batch_size, 1, height, width)
    return adv_images.detach(), full_mask


def run_experiment(
    config,
    transform,
    model,
    device,
    selected_images,
    attack_output_dir,
    attack_cls,
):

    attack = attack_cls(model, config)

    successful_attacks = 0
    ssim_list = []
    psnr_list = []
    modified_pixel_counts = []
    modified_pixel_ratios = []
    perturbation_means = []
    saved_adv_count = 0
    
    # 无目标攻击
    ssim_counts = {0.980: 0, 0.985: 0, 0.990: 0, 0.995: 0, 0.999: 0}
    psnr_counts = {42: 0, 44: 0, 46: 0, 48: 0, 50: 0}

    # 有目标攻击
    # ssim_counts = {0.970: 0,0.975: 0, 0.980: 0, 0.985: 0, 0.990: 0}
    # psnr_counts = {37: 0, 39: 0, 41: 0, 43: 0, 45: 0}

    for img_name in tqdm(selected_images, desc=config["attack_name"]):
        img_path = os.path.join(config["val_dir"], img_name)
        original_img_pil = Image.open(img_path).convert("RGB")

        input_tensor = transform(original_img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            original_output = model(input_tensor)
            original_pred = torch.argmax(original_output, dim=1).item()

        labels = torch.tensor([original_pred]).to(device)

        if config["if_target"]:
            adv_images = attack(input_tensor, target_labels=config["target_labels"])
        else:
            adv_images = attack(input_tensor, labels=labels)

        if config["if_prune"]:
            adv_images_pruned, final_mask = cumulative_gradient_pruning(
                model=model,
                adv_images=adv_images,
                clean_images=input_tensor,
                labels=config["target_labels"] if attack.targeted else labels,
                targeted=attack.targeted,
                step_ratio=config["step_ratio"],
                max_ratio=config["max_ratio"],
            )
            adv_images = adv_images_pruned

        with torch.no_grad():
            adv_output = model(adv_images)
            adv_pred = torch.argmax(adv_output, dim=1).item()

        if_attack_success = True
        if config["if_target"]:
            if adv_pred == config["target_labels"].item():
                successful_attacks += 1
            else:
                if_attack_success = False
        else:
            if original_pred != adv_pred:
                successful_attacks += 1
            else:
                if_attack_success = False

        if if_attack_success:
            adv_img = adv_images.squeeze().cpu().permute(1, 2, 0).numpy()
            ori_img = input_tensor.squeeze().cpu().permute(1, 2, 0).numpy()

            delta = adv_images - input_tensor
            perturbation_mean = torch.mean(torch.abs(delta)).item()
            perturbation_means.append(perturbation_mean)

            ssim_val = ssim(ori_img, adv_img, channel_axis=2, data_range=1.0)
            psnr_val = psnr(ori_img, adv_img, data_range=1.0)

            ssim_list.append(ssim_val)
            psnr_list.append(psnr_val)

            for thresh in ssim_counts:
                if ssim_val >= thresh:
                    ssim_counts[thresh] += 1
            for thresh in psnr_counts:
                if psnr_val >= thresh:
                    psnr_counts[thresh] += 1

            mod_pixels, total_pixels = count_modified_pixels(
                adv_images,
                input_tensor,
                threshold=config["threshold"],
            )
            modified_pixel_counts.append(mod_pixels)
            modified_pixel_ratios.append(mod_pixels / total_pixels)

            if config["if_save_adv"] and saved_adv_count < config["MAX_SAVE_ADV"]:
                adv_img_pil = Image.fromarray((adv_img * 255).astype(np.uint8))
                adv_img_pil.save(os.path.join(attack_output_dir, f"adv_{img_name}"))
                saved_adv_count += 1

    success_rate = successful_attacks / len(selected_images) * 100
    avg_ssim = np.mean(ssim_list) if ssim_list else 0

    valid_psnr_list = [value for value in psnr_list if not np.isinf(value)]
    avg_psnr = np.mean(valid_psnr_list) if valid_psnr_list else 0

    avg_modified_pixels = (
        np.mean(modified_pixel_counts) if modified_pixel_counts else 0
    )
    avg_modified_ratio = np.mean(modified_pixel_ratios) if modified_pixel_ratios else 0
    avg_perturbation_mean = np.mean(perturbation_means) if perturbation_means else 0

    print(f"配置: {config}")
    print(f"成功攻击: {successful_attacks}/{len(selected_images)}")
    print(f"攻击成功率: {success_rate:.2f}%")
    print(f"平均被修改像素点数量: {avg_modified_pixels:.2f}")
    print(f"平均被修改像素比例: {avg_modified_ratio * 100:.2f}%")
    print(f"平均扰动均值: {avg_perturbation_mean*1000000:.2f}")
    print(f"平均 SSIM: {avg_ssim:.6f}")
    print(f"平均 PSNR: {avg_psnr:.4f}dB")

    for thresh in sorted(ssim_counts.keys()):
        percentage = (
            ssim_counts[thresh] / successful_attacks * 100
            if successful_attacks > 0
            else 0
        )
        print(f"SSIM >= {thresh}: {percentage:.2f}%")
    for thresh in sorted(psnr_counts.keys()):
        percentage = (
            psnr_counts[thresh] / successful_attacks * 100
            if successful_attacks > 0
            else 0
        )
        print(f"PSNR >= {thresh}dB: {percentage:.2f}%")

    ssim_percentages = {}
    psnr_percentages = {}
    for thresh in ssim_counts:
        ssim_percentages[thresh] = (
            ssim_counts[thresh] / successful_attacks * 100
            if successful_attacks > 0
            else 0
        )
    for thresh in psnr_counts:
        psnr_percentages[thresh] = (
            psnr_counts[thresh] / successful_attacks * 100
            if successful_attacks > 0
            else 0
        )

    return {
        "success_rate": success_rate,
        "avg_ssim": avg_ssim,
        "avg_psnr": avg_psnr,
        "avg_modified_pixels": avg_modified_pixels,
        "avg_modified_ratio": avg_modified_ratio,
        "avg_perturbation_mean": avg_perturbation_mean,
        "ssim_percentages": ssim_percentages,
        "psnr_percentages": psnr_percentages,
    }
