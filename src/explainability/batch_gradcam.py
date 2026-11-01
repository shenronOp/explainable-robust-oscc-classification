
from pathlib import Path
from src.explainability.gradcam import generate_gradcam


def run_batch_gradcam(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    images = sorted(
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in extensions
    )

    if not images:
        raise ValueError(f"No images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for image in images:
        output_path = output_dir / f"{image.stem}_gradcam.png"

        result = generate_gradcam(
            str(image),
            str(output_path)
        )

        print(
            f"{image.name}: "
            f"{result['predicted_class']} "
            f"-> {output_path}"
        )

    print(f"\nGenerated explanations: {len(images)}")


