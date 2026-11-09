
from pathlib import Path

import torch
import timm


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "efficientnet_b0_best.pth"

# RetinaScope configuration
NUM_CLASSES = 5
DEVICE = torch.device("cpu")

CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    print("Loading RetinaScope EfficientNet-B0...")

    # Recreate the exact training architecture
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=NUM_CLASSES
    )

    # Load trained checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully!")
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Validation accuracy:", checkpoint["val_accuracy"])
    print("Validation F1:", checkpoint["val_f1"])

    return model


if __name__ == "__main__":

    model = load_model()

    print("\nRetinaScope model verification successful.")
    print("Device:", DEVICE)
    print("Number of classes:", NUM_CLASSES)
