import base64
import io
import logging
from pathlib import Path
from typing import Optional, Union

import requests
from PIL import Image

from .config import OLLAMA_GENERATE_URL, OLLAMA_VISION_MODEL

logger = logging.getLogger(__name__)


def image_to_base64(image: Union[Image.Image, Path, str]) -> str:
    """Convert a PIL Image or image file path to a base64 string."""
    if isinstance(image, (str, Path)):
        image = Image.open(str(image))

    # Convert to RGB to avoid alpha channel issues with some VLMs
    if image.mode != "RGB":
        image = image.convert("RGB")

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def analyze_engineering_drawing(
    image: Union[Image.Image, Path, str], 
    prompt: Optional[str] = None
) -> str:
    """
    Pass an engineering drawing (P&ID, schematic) or photo to the local Vision model (e.g., llava).
    Returns a textual description to be indexed in the RAG system.
    """
    if not prompt:
        prompt = (
            "You are an expert industrial engineer. Describe this image in detail. "
            "If it is a Piping and Instrumentation Diagram (P&ID), identify all "
            "equipment tags, valves, pipelines, and their connections. If it is a "
            "photo of equipment, describe its condition, type, and any visible defects or labels. "
            "Extract all text, numbers, and technical specifications exactly as they appear."
        )

    try:
        b64_image = image_to_base64(image)
        
        payload = {
            "model": OLLAMA_VISION_MODEL,
            "prompt": prompt,
            "images": [b64_image],
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }
        
        logger.info(f"Sending image to local VLM: {OLLAMA_VISION_MODEL}")
        response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=300)
        response.raise_for_status()
        
        data = response.json()
        description = data.get("response", "").strip()
        
        if description:
            logger.info("Successfully received description from VLM.")
            return f"[VISION MODEL DESCRIPTION]\n{description}"
        else:
            logger.warning("VLM returned an empty description.")
            return ""

    except Exception as e:
        logger.warning(f"Failed to analyze image with VLM ({OLLAMA_VISION_MODEL}): {e}")
        return ""
