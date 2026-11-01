
import torch
from PIL import Image
from torchvision import transforms
from src.models.efficientnet import create_model

CLASSES = [
    "MD-OSCC",
    "Normal",
    "OSMF",
    "PD-OSCC",
    "WD-OSCC"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = create_model(num_classes=5)

checkpoint = torch.load(
    "models/orchid_experiment_best.pth",
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probabilities, dim=1)

    return {
        "predicted_class": CLASSES[predicted.item()],
        "confidence": round(confidence.item() * 100, 2),
        "probabilities": {
            name: round(prob.item() * 100, 2)
            for name, prob in zip(CLASSES, probabilities[0])
        }
    }
