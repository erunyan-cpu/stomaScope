# core/model_loader.py
import torch
import segmentation_models_pytorch as smp
import warnings
import os
import sys

MODEL_RELATIVE_PATH = "model/best_stomata_model.pth"


def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and PyInstaller.
    """
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def get_device():
    """
    Robust CUDA detection that won’t crash frozen apps.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        # Force a minimal CUDA call to ensure runtime is valid
        torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(0)

        # Warn for very new architectures
        if capability[0] * 10 + capability[1] > 90:
            warnings.warn(
                f"GPU {torch.cuda.get_device_name(0)} "
                f"(sm_{capability[0]}{capability[1]}) may not be fully supported."
            )

        return torch.device("cuda")

    except Exception as e:
        warnings.warn(f"CUDA detected but unusable: {e}. Falling back to CPU.")
        return torch.device("cpu")


def load_model():
    device = get_device()

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # safer for packaged builds
        in_channels=1,
        classes=1,
        activation=None
    )

    model_path = resource_path(MODEL_RELATIVE_PATH)

    state_dict = torch.load(model_path, map_location=device)

    # Remove DataParallel prefix if present
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, device
