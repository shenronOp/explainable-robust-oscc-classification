
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.inference.predict import model, transform, DEVICE, CLASSES


def generate_gradcam(image_path, output_path, target_class=None):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    activations = []
    gradients = []

    layer = model.features[-1]

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = layer.register_forward_hook(forward_hook)
    h2 = layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad()

        output = model(input_tensor)
        predicted = output.argmax(dim=1).item()

        if target_class is None:
            target = predicted
        else:
            if target_class not in CLASSES:
                raise ValueError(f"Invalid target class. Choose from: {CLASSES}")
            target = CLASSES.index(target_class)

        output[0, target].backward()

        acts = activations[0]
        grads = gradients[0]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze()
        cam = torch.relu(cam)

        cam -= cam.min()
        cam /= cam.max() + 1e-8

        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        ).squeeze()

        original = np.array(image.resize((224, 224)))
        heatmap = cam.detach().cpu().numpy()

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(original)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap="jet")
        plt.title(f"Grad-CAM: {CLASSES[target]}")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(original)
        plt.imshow(heatmap, cmap="jet", alpha=0.4)
        plt.title("Overlay")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()

        return {
            "predicted_class": CLASSES[predicted],
            "explained_class": CLASSES[target],
            "output": output_path
        }

    finally:
        h1.remove()
        h2.remove()
