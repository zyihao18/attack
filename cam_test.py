from __future__ import annotations

import argparse
import random
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchcam.metrics import ClassificationMetric
from torchcam.methods import GradCAM, GradCAMpp, LayerCAM, ScoreCAM, SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.models import get_model, get_model_weights
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
CAM_METHODS = {
    "scorecam": ("ScoreCAM", ScoreCAM, False),
    "gradcam": ("GradCAM", GradCAM, True),
    "gradcampp": ("GradCAMpp", GradCAMpp, True),
    "smoothgradcampp": ("SmoothGradCAMpp", SmoothGradCAMpp, True),
    "layercam": ("LayerCAM", LayerCAM, True),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def split_csv(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def resolve_methods(value: str) -> list[str]:
    methods = split_csv(value)
    if len(methods) == 1 and methods[0] == "all":
        return list(CAM_METHODS)

    unknown = [method for method in methods if method not in CAM_METHODS]
    if unknown:
        available = ", ".join(CAM_METHODS)
        raise ValueError(f"Unknown CAM method(s) {unknown}. Available methods: {available}")
    return methods


def list_images(val_dir: Path) -> list[Path]:
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation image directory does not exist: {val_dir}")
    return sorted(
        path
        for path in val_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def select_images(val_dir: Path, count: int) -> list[Path]:
    images = list_images(val_dir)
    if count <= 0 or count >= len(images):
        return images
    return random.sample(images, count)


def load_model(model_name: str, device: torch.device):
    weights = get_model_weights(model_name).DEFAULT
    model = get_model(model_name, weights=weights).to(device)
    model.eval()
    return model, weights, weights.transforms()


def load_rgb_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def draw_label(
    image: Image.Image,
    class_name: str,
    probability: float,
    font_path: str | None,
    font_size: int,
) -> None:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(font_path or "arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    draw.text((1, 1), f"{class_name}\n{probability:.4f}", fill="red", font=font)


def visualize_and_save(
    image: Image.Image,
    cam_map_tensor: torch.Tensor,
    class_name: str,
    probability: float,
    save_path: Path,
    image_size: int,
    alpha: float,
    font_path: str | None,
    font_size: int,
) -> None:
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    cam_map = cam_map_tensor.detach().cpu().squeeze()
    cam_pil = to_pil_image(cam_map, mode="F").resize(
        (image_size, image_size),
        resampling,
    )
    base_pil = image.resize((image_size, image_size), resampling)
    result = overlay_mask(base_pil, cam_pil, alpha=alpha)
    draw_label(result, class_name, probability, font_path, font_size)
    result.save(save_path)


def run_cam_method(
    method_key: str,
    image_paths: list[Path],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    display_name, extractor_cls, requires_grad = CAM_METHODS[method_key]
    save_dir = Path(args.output_dir) / display_name
    save_dir.mkdir(parents=True, exist_ok=True)

    model, weights, preprocess = load_model(args.model, device)
    extractor = extractor_cls(model)
    metrics = ClassificationMetric(extractor, partial(torch.softmax, dim=-1))

    for image_path in tqdm(image_paths, desc=display_name):
        image = load_rgb_image(image_path)
        x = preprocess(image).to(device)
        x_batch = x.unsqueeze(0)

        if requires_grad:
            x_batch.requires_grad_(True)
            output = model(x_batch)
        else:
            with torch.no_grad():
                output = model(x_batch)

        pred_idx = output.squeeze(0).argmax().item()
        class_name = weights.meta["categories"][pred_idx]
        probability = F.softmax(output, dim=1)[0, pred_idx].item()
        activation_map = extractor(pred_idx, output)
        cam_map = activation_map[0].squeeze(0).detach().cpu()

        visualize_and_save(
            image=image,
            cam_map_tensor=cam_map,
            class_name=class_name,
            probability=probability,
            save_path=save_dir / image_path.name,
            image_size=args.image_size,
            alpha=args.alpha,
            font_path=args.font_path,
            font_size=args.font_size,
        )

        metrics.update(x_batch.detach())
        model.zero_grad(set_to_none=True)

    summary = metrics.summary()
    print(f"{display_name} results:")
    print(summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CAM visualizations.")
    parser.add_argument("--val-dir", default="ILSVRC2012_img_val")
    parser.add_argument("--output-dir", default="ImageNet_CAM_swin")
    parser.add_argument("--model", default="resnet34")
    parser.add_argument("--methods", default="all", help="CAM methods, comma-separated, or 'all'.")
    parser.add_argument("--count", type=int, default=50, help="Number of images; <=0 means all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha.")
    parser.add_argument("--font-path", default=None)
    parser.add_argument("--font-size", type=int, default=22)
    return parser


def main() -> int:
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    image_paths = select_images(Path(args.val_dir), args.count)
    print(f"Device: {device}")
    print(f"Selected images: {len(image_paths)}")

    for method_key in resolve_methods(args.methods):
        run_cam_method(method_key, image_paths, args, device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
