from __future__ import annotations

import argparse
import importlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms

from attack_utils import run_experiment


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


@dataclass(frozen=True)
class ModelSpec:
    constructor_name: str
    weights: Any
    dataset_dir: str


@dataclass(frozen=True)
class AttackSpec:
    module_name: str
    class_name: str
    defaults: dict[str, Any]


MODEL_SPECS: dict[str, ModelSpec] = {
    "alexnet": ModelSpec(
        "alexnet",
        models.AlexNet_Weights.IMAGENET1K_V1,
        "correct_1000_AlexNet",
    ),
    "densenet121": ModelSpec(
        "densenet121",
        models.DenseNet121_Weights.IMAGENET1K_V1,
        "correct_1000_DenseNet121",
    ),
    "googlenet": ModelSpec(
        "googlenet",
        models.GoogLeNet_Weights.IMAGENET1K_V1,
        "correct_1000_GoogLeNet",
    ),
    "mobilenet_v3_large": ModelSpec(
        "mobilenet_v3_large",
        models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        "correct_1000_MobileNetV3_large",
    ),
    "resnet34": ModelSpec(
        "resnet34",
        models.ResNet34_Weights.IMAGENET1K_V1,
        "correct_1000_ResNet34",
    ),
    "vgg11": ModelSpec(
        "vgg11",
        models.VGG11_Weights.IMAGENET1K_V1,
        "correct_1000_VGG11",
    ),
    "efficientnet_b0": ModelSpec(
        "efficientnet_b0",
        models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "correct_1000_EfficientNet_b0",
    ),
}


BASE_CONFIG: dict[str, Any] = {
    "SEED": 42,
    "selected_count": 10,
    "output_dir": "adversarial_samples",
    "MAX_SAVE_ADV": 10,
    "if_save_adv": True,
    "threshold": 1e-6,
    "if_target": False,
    "if_prune": False,
    "step_ratio": 0.01,
    "max_ratio": 1.0,
}


ATTACK_SPECS: dict[str, AttackSpec] = {
    "pgd": AttackSpec(
        "impl_pgd",
        "PGD",
        {
            "attack_name": "pgd_untarget",
            "eps": 100 / 255,
            "alpha": 1 / 255,
            "steps": 100,
            "random_start": False,
            "targeted": False,
            "if_target": False,
        },
    ),
    "mifgsm": AttackSpec(
        "impl_mifgsm",
        "MIFGSM",
        {
            "attack_name": "mifgsm_target",
            "eps": 100 / 255,
            "alpha": 1 / 255,
            "steps": 100,
            "decay": 1.0,
            "targeted": True,
            "if_target": True,
        },
    ),
    "cw": AttackSpec(
        "impl_cw",
        "CWLinf",
        {
            "attack_name": "cw_untarget",
            "eps": 100 / 255,
            "alpha": 1 / 255,
            "steps": 100,
            "random_start": False,
            "targeted": False,
            "if_target": False,
            "kappa": 0.0,
        },
    ),
    "apgd_ce": AttackSpec(
        "impl_apgd_ce",
        "APGD_CE_Linf",
        {
            "attack_name": "apgd_ce_untarget",
            "eps": 100 / 255,
            "alpha": 1 / 255,
            "steps": 100,
            "random_start": False,
            "targeted": False,
            "if_target": False,
            "n_restarts": 2,
            "rho": 0.75,
            "eot_iter": 1,
            "early_stop": True,
        },
    ),
    "apgd_dlr": AttackSpec(
        "impl_apgd_dlr",
        "APGD_DLR_Linf",
        {
            "attack_name": "apgd_dlr_untarget",
            "eps": 100 / 255,
            "alpha": 1 / 255,
            "steps": 100,
            "random_start": False,
            "targeted": False,
            "if_target": False,
            "n_restarts": 2,
            "rho": 0.75,
            "eot_iter": 1,
            "early_stop": True,
        },
    ),
    "mymodel": AttackSpec(
        "impl_myattack",
        "MyAttack",
        {
            "attack_name": "SRGA_CGP_target",
            "cam_keep_ratio": 0.45,
            "eps": 100 / 255,
            "alpha": 1 / 255,
            "steps": 100,
            "random_start": False,
            "targeted": True,
            "if_target": True,
            "if_prune": True,
        },
    ),
}


ATTACK_ALIASES = {
    "myattack": "mymodel",
    "srga_cgp": "mymodel",
}


def parse_float_expr(value: str) -> float:
    value = str(value).strip()
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def parse_optional_float(value: str) -> float | None:
    if str(value).strip().lower() in {"none", "null"}:
        return None
    return parse_float_expr(value)


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def add_optional_bool(
    parser: argparse.ArgumentParser,
    name: str,
    dest: str,
    help_text: str,
) -> None:
    parser.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    parser.add_argument(
        f"--no-{name}",
        dest=dest,
        action="store_false",
        help=f"Disable {help_text.lower()}",
    )
    parser.set_defaults(**{dest: None})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_attack_key(attack_key: str) -> str:
    normalized = attack_key.strip().lower()
    normalized = ATTACK_ALIASES.get(normalized, normalized)
    if normalized not in ATTACK_SPECS:
        available = ", ".join(sorted(ATTACK_SPECS))
        raise ValueError(f"Unknown attack '{attack_key}'. Available attacks: {available}")
    return normalized


def resolve_attack_keys(value: str) -> list[str]:
    keys = split_csv(value)
    if not keys:
        raise ValueError("At least one attack must be specified.")
    if len(keys) == 1 and keys[0].lower() == "all":
        return list(ATTACK_SPECS)
    return [resolve_attack_key(key) for key in keys]


def resolve_model_keys(value: str) -> list[str]:
    keys = split_csv(value)
    if not keys:
        raise ValueError("At least one model must be specified.")
    if len(keys) == 1 and keys[0].lower() == "all":
        return list(MODEL_SPECS)

    normalized_keys = [key.lower() for key in keys]
    unknown = [key for key in normalized_keys if key not in MODEL_SPECS]
    if unknown:
        available = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model(s) {unknown}. Available models: {available}")
    return normalized_keys


def load_attack_class(attack_key: str) -> type:
    spec = ATTACK_SPECS[attack_key]
    module = importlib.import_module(spec.module_name)
    return getattr(module, spec.class_name)


def build_model(model_key: str, device: torch.device) -> torch.nn.Module:
    spec = MODEL_SPECS[model_key]
    constructor = getattr(models, spec.constructor_name)
    model = constructor(weights=spec.weights).to(device)
    model.eval()
    return model


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
        ]
    )


def list_images(dataset_dir: Path) -> list[str]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    return sorted(
        path.name
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def select_images(dataset_dir: Path, selected_count: int) -> list[str]:
    all_images = list_images(dataset_dir)
    if selected_count <= 0 or selected_count >= len(all_images):
        return all_images
    return random.sample(all_images, selected_count)


def build_config(
    args: argparse.Namespace,
    attack_key: str,
    device: torch.device,
) -> dict[str, Any]:
    attack_spec = ATTACK_SPECS[attack_key]
    config = dict(BASE_CONFIG)
    config.update(attack_spec.defaults)

    config.update(
        {
            "SEED": args.seed,
            "selected_count": args.selected_count,
            "output_dir": args.output_dir,
            "MAX_SAVE_ADV": args.max_save_adv,
            "if_save_adv": args.save_adv,
            "threshold": args.threshold,
            "target_labels": torch.tensor([args.target_label], device=device),
            "step_ratio": args.step_ratio,
            "max_ratio": args.max_ratio,
        }
    )

    scalar_overrides = {
        "eps": args.eps,
        "alpha": args.alpha,
        "steps": args.steps,
        "decay": args.decay,
        "kappa": args.kappa,
        "cam_keep_ratio": args.cam_keep_ratio,
        "n_restarts": args.n_restarts,
        "rho": args.rho,
        "eot_iter": args.eot_iter,
    }
    for key, value in scalar_overrides.items():
        if value is not None:
            config[key] = value

    if args.attack_name:
        config["attack_name"] = args.attack_name
    if args.targeted is not None:
        config["targeted"] = args.targeted
        config["if_target"] = args.targeted
    if args.prune is not None:
        config["if_prune"] = args.prune
    if args.random_start is not None:
        config["random_start"] = args.random_start
    if args.early_stop is not None:
        config["early_stop"] = args.early_stop

    return config


def get_dataset_dir(
    args: argparse.Namespace,
    model_key: str,
    model_count: int,
) -> Path:
    if args.dataset_dir:
        if model_count > 1:
            raise ValueError("--dataset-dir can only be used with a single model.")
        return Path(args.dataset_dir)
    return Path(args.dataset_root) / MODEL_SPECS[model_key].dataset_dir


def run_attack(
    args: argparse.Namespace,
    attack_key: str,
    model_keys: list[str],
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    set_seed(args.seed)
    attack_cls = load_attack_class(attack_key)
    results: dict[str, dict[str, Any]] = {}

    for model_key in model_keys:
        run_config = build_config(args, attack_key, device)
        dataset_dir = get_dataset_dir(args, model_key, len(model_keys))
        selected_images = select_images(dataset_dir, run_config["selected_count"])
        if not selected_images:
            print(f"Skip {attack_key}/{model_key}: no images found in {dataset_dir}")
            continue

        output_attack_name = run_config["attack_name"]
        if len(model_keys) > 1 and not args.attack_name:
            output_attack_name = f"{output_attack_name}_{model_key}"
            run_config["attack_name"] = output_attack_name

        attack_output_dir = Path(run_config["output_dir"]) / output_attack_name
        attack_output_dir.mkdir(parents=True, exist_ok=True)

        run_config["val_dir"] = str(dataset_dir)

        print(f"\n=== Start: attack={attack_key}, model={model_key} ===")
        print(f"Dataset: {dataset_dir}")
        print(f"Selected images: {len(selected_images)}")
        print(f"Output: {attack_output_dir}")

        model = build_model(model_key, device)
        result = run_experiment(
            config=run_config,
            transform=build_transform(),
            model=model,
            device=device,
            selected_images=selected_images,
            attack_output_dir=str(attack_output_dir),
            attack_cls=attack_cls,
        )
        results[f"{attack_key}:{model_key}"] = result
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"=== Done: attack={attack_key}, model={model_key} ===")

    return results


def build_parser(default_attack: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run adversarial attack experiments.")
    parser.add_argument(
        "--attack",
        default=default_attack or "pgd",
        help="Attack name, comma-separated names, or 'all'.",
    )
    parser.add_argument(
        "--models",
        default="resnet34",
        help="Model name, comma-separated names, or 'all'.",
    )
    parser.add_argument("--dataset-root", default=".", help="Root containing correct_1000_* folders.")
    parser.add_argument("--dataset-dir", default=None, help="Custom dataset directory for one model.")
    parser.add_argument("--output-dir", default="adversarial_samples", help="Output root.")
    parser.add_argument("--attack-name", default=None, help="Override output subdirectory name.")
    parser.add_argument("--selected-count", type=int, default=10, help="Number of images; <=0 means all.")
    parser.add_argument("--max-save-adv", type=int, default=10, help="Maximum adversarial images to save.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--target-label", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--step-ratio", type=float, default=0.01)
    parser.add_argument("--max-ratio", type=float, default=1.0)

    parser.add_argument("--eps", type=parse_float_expr, default=None)
    parser.add_argument("--alpha", type=parse_optional_float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--decay", type=float, default=None)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--cam-keep-ratio", type=float, default=None)
    parser.add_argument("--n-restarts", type=int, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--eot-iter", type=int, default=None)

    parser.add_argument("--save-adv", dest="save_adv", action="store_true", default=True)
    parser.add_argument("--no-save-adv", dest="save_adv", action="store_false")
    add_optional_bool(parser, "targeted", "targeted", "targeted mode")
    add_optional_bool(parser, "prune", "prune", "cumulative gradient pruning")
    add_optional_bool(parser, "random-start", "random_start", "random start")
    add_optional_bool(parser, "early-stop", "early_stop", "early stopping")

    parser.add_argument("--list-attacks", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    return parser


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main(default_attack: str | None = None, argv: list[str] | None = None) -> int:
    parser = build_parser(default_attack=default_attack)
    args = parser.parse_args(argv)

    if args.list_attacks:
        print("\n".join(sorted(ATTACK_SPECS)))
        return 0
    if args.list_models:
        print("\n".join(sorted(MODEL_SPECS)))
        return 0

    attack_keys = resolve_attack_keys(args.attack)
    model_keys = resolve_model_keys(args.models)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    for attack_key in attack_keys:
        run_attack(args, attack_key, model_keys, device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
