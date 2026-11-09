
import cv2
import numpy as np
import torch


def preprocess_image(image):
    image = np.array(
        image.convert("RGB")
    )

    # Resize
    image = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    # CLAHE
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(
        l_channel
    )

    lab = cv2.merge(
        [l_channel, a_channel, b_channel]
    )

    image = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    # Light denoising
    image = cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )

    # Visualization image
    rgb_image = image.astype(
        np.float32
    ) / 255.0

    # Tensor
    tensor = torch.from_numpy(
        rgb_image
    ).permute(2, 0, 1).float()

    # ImageNet normalization
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    tensor = (tensor - mean) / std

    tensor = tensor.unsqueeze(0)

    return rgb_image, tensor
