
import csv
from pathlib import Path
from src.inference.predict import predict_image


def run_batch(input_dir, output_file):
    input_dir = Path(input_dir)

    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    images = sorted(
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in extensions
    )

    if not images:
        raise ValueError(f"No images found in {input_dir}")

    rows = []

    for image in images:
        result = predict_image(str(image))

        rows.append({
            "image": str(image),
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            **result["probabilities"]
        })

        print(
            f"{image.name}: "
            f"{result['predicted_class']} "
            f"({result['confidence']}%)"
        )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nProcessed images: {len(rows)}")
    print(f"Results saved to: {output_file}")

    return rows
