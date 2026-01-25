"""
Minimal multimodal call using the llms interface with interleaved images/text.
"""

import json
import os
from pathlib import Path

from PIL import Image as PILImage

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from llms import load_model


def run_test(model, images) -> None:
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
    print("response =================================")
    print(response)

    prompts = [
        (
        "You will see one image, interleaved with text labels.",
        "Image 1:",
        images[0],
        "Describe the image at the end, labeled Image 1, in 1-2 sentences.",
    ),
    (
        "You will see two images, interleaved with text labels.",
        "Image 1:",
        images[0],
        "Image 2:",
        images[1],
        "Describe the images at the end, labeled Image 1/2, in 1-2 sentences each.",
    )
        # ("Image 1", images[0], "Image 2", images[1], "what are in this image 1 and 2 separately?"),
        # ("Image 3", images[2], "what is in this image 3?"),
    ]
    responses = model(prompts)
    print("responses batched ", responses)
    for idx, text in enumerate(responses, start=1):
        print(f"\n--- Batch response {idx} ---\n{text}")

    print("response separate =================================")
    response = model(prompts[0])
    print("1")
    print(response)
    response = model(prompts[1])
    print("2")
    print(response)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    keys_path = repo_root / "API_KEYS2.json"
    with keys_path.open("r") as file:
        api_keys = json.load(file)

    os.environ["OPENAI_API_KEY"] = api_keys["OPENAI_API_KEY"]
    os.environ["ANTHROPIC_API_KEY"] = api_keys["ANTHROPIC_API_KEY"]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
        repo_root / api_keys["GOOGLE_APPLICATION_CREDENTIALS"]
    )
    # os.environ.setdefault("VERTEX_PROJECT_ID", "tfix-485319")
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

    models = [
        #"gpt-5.2-pro-2025-12-11",
        #"gpt-5-mini-2025-08-07",
        "gpt-5-nano",
        # "claude-opus-4-5-20251101",
        "claude-haiku-4-5-20251001",
        # "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]
    for model_name in models:
        print(f"\n=== Testing {model_name} ===")
        model = load_model(model_name)
        run_test(model, images)


if __name__ == "__main__":
    main()
