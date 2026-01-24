"""
Minimal multimodal call using the llms interface with interleaved images/text.
"""

import os
from pathlib import Path

from PIL import Image as PILImage

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from llms import load_model


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adc_path = repo_root / "application_default_credentials.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(adc_path)
    os.environ.setdefault("VERTEX_PROJECT_ID", "surgery-483823")
    os.environ.setdefault("VERTEX_LOCATION", "us-central1")

    image_paths = [
        repo_root / "src" / "prompts" / "data" / "cholec_fewshot_1_image.png",
        repo_root / "src" / "prompts" / "data" / "cholec_fewshot_1_safe.png",
        repo_root / "src" / "prompts" / "data" / "cholec_fewshot_1_unsafe.png",
    ]
    for path in image_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing image file: {path}")

    images = [PILImage.open(p) for p in image_paths]

    model = load_model("gemini-2.0-flash")
    prompt = (
        "You will see three images, interleaved with text labels.",
        "Image 1:",
        images[0],
        "Image 2:",
        images[1],
        "Image 3:",
        images[2],
        "Describe each image at the end, labeled Image 1/2/3, in 1-2 sentences each.",
    )
    response = model(prompt)
    print(response)

    prompts = [
        ("Image 1", images[0], "Image 2", images[1], "what are in image 1 and 2 separately?"),
        ("Image 3", images[2], "what is in image 3?"),
    ]
    responses = model(prompts)
    for idx, text in enumerate(responses, start=1):
        print(f"\n--- Batch response {idx} ---\n{text}")


if __name__ == "__main__":
    main()
