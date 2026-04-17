"""
Reads a CoT + action text file and uses a Vertex AI image generation model
to generate an image containing that text.

Supports:
  - Gemini image generation models (e.g. gemini-2.5-flash-image)
  - Imagen models (e.g. imagen-4.0-generate-001)

Usage:
    python generate_image.py [input_file] [model_id] [aspect_ratio]

    input_file    — path to CoT text file (default: sample_cot.txt)
    model_id      — full or short model ID (default: publishers/google/models/gemini-2.5-flash-image)
    aspect_ratio  — e.g. 1:1, 16:9, 9:16, 4:3 (default: 1:1)

Output image is named automatically: output_<model-name>.png

Examples:
    python generate_image.py
    python generate_image.py sample_cot.txt publishers/google/models/gemini-2.5-flash-image 16:9
    python generate_image.py sample_cot.txt publishers/google/models/imagen-4.0-generate-001 9:16

Required env vars (source api_setup.sh from the webarena project):
    PROJECT_ID  — GCP project ID
    LOCATION    — GCP region (e.g. us-central1)
"""

import os
import sys
import base64
import warnings
from pathlib import Path

# Suppress upstream FutureWarnings about Python 3.10 EOL and google-cloud-storage
# version compatibility — informational only, not actionable right now.
warnings.filterwarnings("ignore", category=FutureWarning, module="google")

import yaml
from google import genai
from google.genai import types

DEFAULT_MODEL = "publishers/google/models/gemini-2.5-flash-image"
PROMPTS_FILE = Path(__file__).parent / "prompts.yaml"


def load_prompts() -> dict:
    with open(PROMPTS_FILE, "r") as f:
        return yaml.safe_load(f)


def read_cot_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read().strip()


def model_short_name(model_id: str) -> str:
    """Extract the last path component for use in the output filename."""
    return model_id.rstrip("/").split("/")[-1]


def output_filename(model_id: str, aspect_ratio: str) -> str:
    ratio_str = aspect_ratio.replace(":", "x")
    return f"output_{model_short_name(model_id)}_{ratio_str}.png"


def generate_with_gemini(client: genai.Client, text: str, model_name: str, output_path: str, prompts: dict, aspect_ratio: str) -> None:
    prompt = prompts["gemini"].format(text=text)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
        ),
    )

    image_saved = False
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            mime = part.inline_data.mime_type
            data = part.inline_data.data
            if isinstance(data, str):
                data = base64.b64decode(data)
            with open(output_path, "wb") as f:
                f.write(data)
            print(f"Image saved to: {output_path}  (mime: {mime})")
            image_saved = True
            break

    if not image_saved:
        print("No image part found in response. Model output:")
        for part in response.candidates[0].content.parts:
            if part.text:
                print(part.text)


def generate_with_imagen(client: genai.Client, text: str, model_name: str, output_path: str, prompts: dict, aspect_ratio: str) -> None:
    prompt = prompts["imagen"].format(text=text)
    response = client.models.generate_images(
        model=model_name,
        prompt=prompt,
        config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio=aspect_ratio),
    )
    image_bytes = response.generated_images[0].image.image_bytes
    with open(output_path, "wb") as f:
        f.write(image_bytes)
    print(f"Image saved to: {output_path}")


def generate_image(text: str, model_id: str, aspect_ratio: str) -> None:
    project = os.environ["PROJECT_ID"]
    location = os.environ.get("LOCATION", "us-central1")
    client = genai.Client(vertexai=True, project=project, location=location)

    # Strip the publishers/google/models/ prefix for the SDK call
    model_name = model_id.rstrip("/").split("/")[-1]
    output_path = output_filename(model_id, aspect_ratio)

    print(f"Model: {model_name}")
    print(f"Aspect ratio: {aspect_ratio}")
    print(f"Output: {output_path}")

    prompts = load_prompts()

    if model_name.startswith("imagen"):
        generate_with_imagen(client, text, model_name, output_path, prompts, aspect_ratio)
    else:
        generate_with_gemini(client, text, model_name, output_path, prompts, aspect_ratio)


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "sample_cot.txt"
    model_id = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    aspect_ratio = sys.argv[3] if len(sys.argv) > 3 else "1:1"

    if not Path(input_path).exists():
        print(f"Error: input file '{input_path}' not found.")
        sys.exit(1)

    text = read_cot_file(input_path)
    print(f"Read {len(text)} characters from '{input_path}'")
    print("--- text ---")
    print(text)
    print("------------")

    generate_image(text, model_id, aspect_ratio)


if __name__ == "__main__":
    main()
