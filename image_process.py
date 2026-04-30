from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


@dataclass(frozen=True)
class ModelSpec:
    output_name: str
    constructor: Callable[..., torch.nn.Module]
    weights: Any


MODEL_SPECS: dict[str, ModelSpec] = {
    "resnet34": ModelSpec("ResNet34", models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "alexnet": ModelSpec("AlexNet", models.alexnet, models.AlexNet_Weights.IMAGENET1K_V1),
    "densenet121": ModelSpec(
        "DenseNet121",
        models.densenet121,
        models.DenseNet121_Weights.IMAGENET1K_V1,
    ),
    "vgg11": ModelSpec("VGG11", models.vgg11, models.VGG11_Weights.IMAGENET1K_V1),
    "googlenet": ModelSpec(
        "GoogLeNet",
        models.googlenet,
        models.GoogLeNet_Weights.IMAGENET1K_V1,
    ),
    "mobilenet_v3_large": ModelSpec(
        "MobileNetV3_large",
        models.mobilenet_v3_large,
        models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    ),
    "efficientnet_b0": ModelSpec(
        "EfficientNet_b0",
        models.efficientnet_b0,
        models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    ),
}


def split_csv(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_optional_int(value: str) -> int | None:
    if value.strip().lower() in {"none", "null"}:
        return None
    return int(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def resolve_models(value: str) -> list[str]:
    keys = split_csv(value)
    if len(keys) == 1 and keys[0] == "all":
        return list(MODEL_SPECS)

    unknown = [key for key in keys if key not in MODEL_SPECS]
    if unknown:
        available = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model(s) {unknown}. Available models: {available}")
    return keys


def load_image_labels(info_path: Path) -> dict[str, int]:
    img_to_label: dict[str, int] = {}
    with info_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            img_to_label[parts[0]] = int(parts[1])
    return img_to_label


def load_label_names(label_path: Path) -> dict[str, Any]:
    with label_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def class_name_for(label_dict: dict[str, Any], label: int) -> str:
    entry = label_dict.get(str(label), str(label))
    if isinstance(entry, list) and len(entry) > 1:
        return str(entry[1])
    return str(entry)


def valid_images(val_img_dir: Path, img_to_label: dict[str, int]) -> list[str]:
    images = [
        path.name
        for path in val_img_dir.iterdir()
        if path.is_file()
        and path.name in img_to_label
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    random.shuffle(images)
    return images


def build_preprocess(weights: Any):
    try:
        return weights.transforms()
    except Exception:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
            ]
        )


def logits_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def collect_correct_images(
    model_key: str,
    image_names: list[str],
    img_to_label: dict[str, int],
    label_dict: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> int:
    spec = MODEL_SPECS[model_key]
    output_prefix = f"correct_{args.target_count}"
    output_dir = Path(args.output_root) / f"{output_prefix}_{spec.output_name}"
    metadata_path = Path(args.metadata_dir) / f"{output_prefix}_{spec.output_name}_metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    model = spec.constructor(weights=spec.weights).to(device).eval()
    preprocess = build_preprocess(spec.weights)

    collected = 0
    skipped_errors = 0
    metadata: list[dict[str, Any]] = []

    with torch.no_grad():
        progress = tqdm(image_names, desc=f"Collect correct images: {spec.output_name}")
        for image_name in progress:
            if collected >= args.target_count:
                break

            true_label = img_to_label[image_name]
            if args.skip_label is not None and true_label == args.skip_label:
                continue

            image_path = Path(args.val_img_dir) / image_name
            try:
                image = Image.open(image_path).convert("RGB")
                tensor = preprocess(image).unsqueeze(0).to(device)
                pred = torch.argmax(logits_from_output(model(tensor)), dim=1).item()
            except Exception:
                skipped_errors += 1
                continue

            if pred != true_label:
                continue

            dst_path = output_dir / image_name
            shutil.copy2(image_path, dst_path)
            collected += 1
            metadata.append(
                {
                    "image_name": image_name,
                    "true_label": true_label,
                    "pred_label": pred,
                    "class_name": class_name_for(label_dict, true_label),
                    "saved_path": str(dst_path.relative_to(Path(args.output_root))),
                    "sequence": collected,
                }
            )
            progress.set_postfix(collected=collected)

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"Done: {spec.output_name}, collected={collected}, skipped_errors={skipped_errors}")
    print(f"Images: {output_dir}")
    print(f"Metadata: {metadata_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return collected


def build_parser() -> argparse.ArgumentParser:
    default_imagenet_dir = Path.cwd().parent / "imagenet"

    parser = argparse.ArgumentParser(description="Collect ImageNet images correctly classified by models.")
    parser.add_argument("--imagenet-dir", default=str(default_imagenet_dir))
    parser.add_argument("--val-img-dir", default=None)
    parser.add_argument("--info-path", default=None)
    parser.add_argument("--label-path", default=None)
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--metadata-dir", default=None)
    parser.add_argument("--models", default="all", help="Model names, comma-separated, or 'all'.")
    parser.add_argument("--target-count", type=int, default=1000)
    parser.add_argument("--skip-label", type=parse_optional_int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def normalize_paths(args: argparse.Namespace) -> None:
    imagenet_dir = Path(args.imagenet_dir)
    args.val_img_dir = str(Path(args.val_img_dir) if args.val_img_dir else imagenet_dir / "ILSVRC2012_img_val")
    args.info_path = str(Path(args.info_path) if args.info_path else imagenet_dir / "imagenet_img_info.txt")
    args.label_path = str(Path(args.label_path) if args.label_path else imagenet_dir / "imagenet_label_english.json")
    args.output_root = str(Path(args.output_root))
    args.metadata_dir = str(Path(args.metadata_dir) if args.metadata_dir else Path(args.output_root) / "metadata")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    normalize_paths(args)

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"ImageNet dir: {args.imagenet_dir}")
    print(f"Validation images: {args.val_img_dir}")
    print(f"Output root: {args.output_root}")

    img_to_label = load_image_labels(Path(args.info_path))
    label_dict = load_label_names(Path(args.label_path))
    image_names = valid_images(Path(args.val_img_dir), img_to_label)
    print(f"Valid candidate images: {len(image_names)}")

    for model_key in resolve_models(args.models):
        collect_correct_images(
            model_key=model_key,
            image_names=image_names,
            img_to_label=img_to_label,
            label_dict=label_dict,
            args=args,
            device=device,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
