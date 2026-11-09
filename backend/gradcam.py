
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_gradcam(model, input_tensor, predicted_class, rgb_image):
    """
    Generate Grad-CAM visualization for the predicted DR class.

    Args:
        model: Loaded RetinaScope EfficientNet-B0 model.
        input_tensor: Preprocessed image tensor [1, 3, 224, 224].
        predicted_class: Predicted class index.
        rgb_image: Preprocessed RGB image with values in [0, 1].

    Returns:
        RGB heatmap visualization as a NumPy array.
    """

    target_layers = [model.conv_head]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    targets = [
        ClassifierOutputTarget(predicted_class)
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_image.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    return visualization
