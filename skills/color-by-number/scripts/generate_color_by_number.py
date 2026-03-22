#!/usr/bin/env python3
"""Generate color-by-number activity pages from a theme and color set.

For every page the script produces:
  - A full-color illustration (<base>-page-NNN-colored.jpg)
  - A matching black-and-white line-art image with numbered regions (<base>-page-NNN-bw.jpg)
  - A Markdown plan file (<base>-plan.md) containing the color key and all prompts used
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime as dt
import json
import mimetypes
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request


CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
IMAGES_URL = "https://api.openai.com/v1/images/generations"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_PROVIDER = "gemini"
DEFAULT_OPENAI_PLANNER_MODEL = "gpt-4o"
DEFAULT_OPENAI_IMAGE_MODEL = "dall-e-3"
DEFAULT_GEMINI_PLANNER_MODEL = "gemini-2.5-pro"
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-3-pro-image-preview"  # Fallback to OpenAI for image generation
DEFAULT_PAGES = 5

# Standard 8-color crayon set used when --colors is not supplied
DEFAULT_COLORS = [
    "Red",
    "Orange",
    "Yellow",
    "Green",
    "Blue",
    "Purple",
    "Brown",
    "Black",
]

# Approximate RGB values for common crayon/color names (used when PIL is unavailable)
COLOR_NAME_MAP: dict[str, tuple[int, int, int]] = {
    "red": (220, 50, 47),
    "orange": (253, 126, 20),
    "yellow": (255, 220, 0),
    "green": (40, 167, 69),
    "blue": (0, 123, 255),
    "purple": (111, 66, 193),
    "brown": (139, 69, 19),
    "black": (30, 30, 30),
    "white": (255, 255, 255),
    "pink": (255, 182, 193),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "light blue": (135, 206, 235),
    "dark blue": (0, 0, 139),
    "sky blue": (135, 206, 235),
    "teal": (0, 128, 128),
    "lime": (50, 205, 50),
    "navy": (0, 0, 128),
    "maroon": (128, 0, 0),
    "olive": (128, 128, 0),
    "tan": (210, 180, 140),
    "gold": (255, 215, 0),
    "silver": (192, 192, 192),
    "beige": (245, 245, 220),
    "coral": (255, 127, 80),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "violet": (238, 130, 238),
    "indigo": (75, 0, 130),
    "turquoise": (64, 224, 208),
    "salmon": (250, 128, 114),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create color-by-number activity pages (colored + B&W numbered versions).",
    )
    parser.add_argument("--theme", help="Theme prompt or description.")
    parser.add_argument(
        "--colors",
        help=(
            "Comma-separated list of color names to use as the palette "
            "(e.g. 'Red,Blue,Green,Yellow'). "
            f"Defaults to the standard {len(DEFAULT_COLORS)}-color crayon set."
        ),
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=DEFAULT_PAGES,
        help=f"Number of pages to generate. Defaults to {DEFAULT_PAGES}.",
    )
    parser.add_argument(
        "--replay-from-md",
        help="Reuse an existing plan Markdown file as editable input.",
    )
    parser.add_argument(
        "--reference-image",
        help="Optional local image path to use as a style reference during generation.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "gemini"),
        help=(
            "API provider to use for planning and image generation. "
            "If omitted, OpenAI is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="tmp",
        help="Directory for output files. Defaults to ./tmp",
    )
    parser.add_argument(
        "--planner-model",
        help="Text model used to plan picture descriptions. Provider-specific default is used if omitted.",
    )
    parser.add_argument(
        "--image-model",
        help="Image model used to render the pages. Provider-specific default is used if omitted.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# API key helpers
# ---------------------------------------------------------------------------

def get_api_key(provider: str) -> str:
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_AI_TOKEN")
        if not api_key:
            raise SystemExit("Missing OpenAI API key. Set OPENAI_API_KEY or OPEN_AI_TOKEN.")
        return api_key

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY.")
    return api_key


def has_gemini_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def default_provider() -> str:
    return "openai"


def get_available_gemini_models(api_key: str) -> list[str]:
    """Return Gemini models that support generateContent."""
    try:
        url = f"{GEMINI_API_BASE_URL}?pageSize=50"
        request = urllib.request.Request(
            url,
            headers={"x-goog-api-key": api_key},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            models = []
            for model_info in data.get("models", []):
                model_name = model_info.get("name", "").replace("models/", "")
                supported_methods = model_info.get("supportedGenerationMethods", [])
                if "generateContent" in supported_methods and "gemini" in model_name.lower():
                    models.append(model_name)
            return models
    except Exception:
        return ["gemini-1.5-flash", "gemini-pro", "gemini-1.0-pro"]


def get_working_gemini_planner_model(api_key: str) -> str:
    """Return the first working Gemini model for planning."""
    available_models = get_available_gemini_models(api_key)
    preferred_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro", "gemini-1.0-pro"]
    for model in preferred_models:
        if model in available_models:
            return model
    if available_models:
        return available_models[0]
    return "gemini-pro"


def default_planner_model(provider: str, api_key: str | None = None) -> str:
    if provider == "gemini":
        if api_key:
            try:
                return get_working_gemini_planner_model(api_key)
            except Exception:
                pass
        return DEFAULT_GEMINI_PLANNER_MODEL
    return DEFAULT_OPENAI_PLANNER_MODEL


def default_image_model(provider: str) -> str:
    if provider == "gemini":
        return DEFAULT_GEMINI_IMAGE_MODEL
    return DEFAULT_OPENAI_IMAGE_MODEL


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def slugify(value: str) -> str:
    slug = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return (slug[:48].rstrip("-")) or "color-by-number"


def guess_mime_type(image_path: pathlib.Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type or not mime_type.startswith("image/"):
        raise SystemExit(f"Unsupported or unknown image type for reference: {image_path}")
    return mime_type


def resolve_reference_image_path(image_path: str | None) -> pathlib.Path | None:
    if not image_path:
        return None
    if image_path.strip() == "-":
        return None
    path = pathlib.Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Reference image not found: {path}")
    return path


def parse_colors(colors_arg: str | None) -> list[str]:
    """Parse comma-separated color list or return the default palette."""
    if not colors_arg:
        return list(DEFAULT_COLORS)
    colors = [c.strip() for c in colors_arg.split(",") if c.strip()]
    if not colors:
        return list(DEFAULT_COLORS)
    return colors


def color_key_text(colors: list[str]) -> str:
    """Return a numbered color-key string, e.g. '1=Red, 2=Orange, ...'"""
    return ", ".join(f"{i + 1}={color}" for i, color in enumerate(colors))


def color_key_markdown(colors: list[str]) -> str:
    """Return a Markdown table for the color key."""
    header = "| Number | Color |\n|--------|-------|\n"
    rows = "".join(f"| {i + 1} | {color} |\n" for i, color in enumerate(colors))
    return header + rows


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def post_json(url: str, payload: dict, api_key: str) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"API request failed: HTTP {exc.code}\n{error_body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Network request failed: {exc.reason}") from exc


def post_gemini_json(model: str, payload: dict, api_key: str) -> dict:
    url = f"{GEMINI_API_BASE_URL}/{model}:generateContent"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Gemini API request failed: HTTP {exc.code}\n{error_body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Network request failed: {exc.reason}") from exc


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def planner_messages(theme: str, colors: list[str], pages: int) -> tuple[str, str]:
    key_text = color_key_text(colors)
    system_prompt = (
        "You are a color-by-number activity book planner. Return valid JSON only. "
        "Create exactly the requested number of distinct picture descriptions for color-by-number pages "
        "that tell a cohesive story with CONSISTENT CHARACTERS. "
        "First identify the main characters from the theme, then create pages that show these same characters in different scenes. "
        "Each page must use ONLY the colors provided in the palette so they can be numbered accordingly. "
        "CRITICAL: Design for BIG, SIMPLE regions that are easy to color. Avoid intricate details. "
        "Do not include markdown fences."
    )
    user_payload = {
        "task": "Plan a printable color-by-number activity book",
        "requirements": {
            "theme": theme,
            "color_palette": colors,
            "color_key": key_text,
            "pages": pages,
            "character_consistency": (
                "identify main characters and ensure they appear with the same visual characteristics across all pages"
            ),
            "storyline": "create a logical progression of scenes that tell a cohesive story",
            "palette_usage": (
                "every scene must be expressible using ONLY the supplied color palette; "
                "describe objects in terms of palette colors; "
                "CRITICAL: each color number must ALWAYS map to the same color - "
                "if Red is #1, then ALL red areas must be #1, never use Red for any other number"
            ),
            "color_consistency": (
                "STRICT RULE: Each palette color must have exactly one number assignment. "
                "No color should appear with different numbers. "
                "If an object is Red (#1), all red areas must be consistently numbered 1"
            ),
            "design_simplicity": (
                "favor LARGE, SIMPLE shapes and regions over intricate details; "
                "each colored area should be big enough to easily fit numbers inside; "
                "minimize small decorative elements"
            ),
            "framing": (
                "ensure all characters, objects, and scene elements are COMPLETELY contained within page boundaries "
                "with comfortable white margins; NO cropped or cut-off elements at any edge"
            ),
            "complexity": "simple design suitable for color-by-number activity for children with big regions to color",
        },
        "output_schema": {
            "theme_title": "short human-readable title for the story",
            "main_characters": [
                "array of 1-3 main characters with detailed physical descriptions for consistency"
            ],
            "picture_descriptions": [
                f"array of exactly {pages} distinct picture description STRINGS (not objects) that advance the story with the same characters using simple, large shapes"
            ],
            "base_prompt": (
                "one detailed style prompt used for all images, emphasizing simple cartoon style with "
                "large colored regions, bold outlines, and minimal details suitable for color-by-number"
            ),
        },
    }
    return system_prompt, json.dumps(user_payload, ensure_ascii=True)


def build_plan_openai(
    api_key: str,
    planner_model: str,
    theme: str,
    colors: list[str],
    pages: int,
) -> tuple[dict, str, str]:
    system_prompt, user_prompt = planner_messages(theme, colors, pages)
    payload = {
        "model": planner_model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    response = post_json(CHAT_COMPLETIONS_URL, payload, api_key)
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SystemExit(f"Unexpected planner response: {json.dumps(response, indent=2)}") from exc

    if not isinstance(content, str):
        raise SystemExit(f"Unexpected planner content format: {json.dumps(response, indent=2)}")

    try:
        plan = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Planner did not return valid JSON:\n{content}") from exc

    picture_descriptions = plan.get("picture_descriptions")
    if not isinstance(picture_descriptions, list):
        raise SystemExit(f"Planner returned invalid picture descriptions:\n{json.dumps(plan, indent=2)}")
    
    # Handle both simple strings and objects with description fields
    processed_descriptions = []
    for item in picture_descriptions:
        if isinstance(item, str) and item.strip():
            processed_descriptions.append(item.strip())
        elif isinstance(item, dict) and "description" in item:
            processed_descriptions.append(item["description"].strip())
        else:
            raise SystemExit(f"Invalid picture description format:\n{json.dumps(item, indent=2)}")
    
    if len(processed_descriptions) != pages:
        raise SystemExit(
            f"Planner returned {len(processed_descriptions)} pictures; expected {pages}."
        )
    
    # Update the plan with processed descriptions
    plan["picture_descriptions"] = processed_descriptions

    main_characters = plan.get("main_characters", [])
    if not isinstance(main_characters, list):
        main_characters = []

    base_prompt = plan.get("base_prompt")
    if not isinstance(base_prompt, str) or not base_prompt.strip():
        char_desc = " Main characters: " + "; ".join(main_characters) if main_characters else ""
        key_text = color_key_text(colors)
        plan["base_prompt"] = (
            f"Simple cartoon style color-by-number activity illustration with LARGE colored regions and bold outlines. "
            f"CRITICAL: Use consistent color-number mapping - each color appears with only one number. "
            f"Minimalist design with big, simple shapes. Color palette: {key_text}.{char_desc}"
        )

    theme_title = plan.get("theme_title")
    if not isinstance(theme_title, str) or not theme_title.strip():
        plan["theme_title"] = theme[:80].strip()

    return plan, system_prompt, user_prompt


def build_plan_gemini(
    api_key: str,
    planner_model: str,
    theme: str,
    colors: list[str],
    pages: int,
) -> tuple[dict, str, str]:
    system_prompt, user_prompt = planner_messages(theme, colors, pages)
    schema = {
        "type": "object",
        "properties": {
            "theme_title": {"type": "string"},
            "main_characters": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5,
            },
            "picture_descriptions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": pages,
                "maxItems": pages,
            },
            "base_prompt": {"type": "string"},
        },
        "required": ["theme_title", "main_characters", "picture_descriptions", "base_prompt"],
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": system_prompt},
                    {"text": user_prompt},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    response = post_gemini_json(planner_model, payload, api_key)
    try:
        content = response["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SystemExit(f"Unexpected Gemini planner response: {json.dumps(response, indent=2)}") from exc

    try:
        plan = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Gemini planner did not return valid JSON:\n{content}") from exc

    return plan, system_prompt, user_prompt


def build_plan(
    provider: str,
    api_key: str,
    planner_model: str,
    theme: str,
    colors: list[str],
    pages: int,
) -> tuple[dict, str, str]:
    if provider == "openai":
        return build_plan_openai(api_key, planner_model, theme, colors, pages)
    return build_plan_gemini(api_key, planner_model, theme, colors, pages)


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def _generate_image_openai(
    api_key: str,
    image_model: str,
    prompt: str,
    reference_image: pathlib.Path | None,
) -> tuple[bytes | None, str | None]:
    full_prompt = prompt
    if reference_image:
        full_prompt = (
            f"{prompt} Use the attached reference image only for style cues such as "
            "line weight, spacing, and composition."
        )
    payload = {
        "model": image_model,
        "prompt": full_prompt,
        "n": 1,
        "size": "1024x1024",
        "response_format": "b64_json",
    }
    try:
        response = post_json(IMAGES_URL, payload, api_key)
        b64_data = response["data"][0]["b64_json"]
        return base64.b64decode(b64_data), None
    except SystemExit as exc:
        error_msg = str(exc)
        if "content_policy_violation" in error_msg or "content filters" in error_msg:
            return None, "Content policy violation - image description may contain restricted content"
        elif "400" in error_msg:
            return None, f"API error: {error_msg}"
        else:
            return None, f"Generation failed: {error_msg}"
    except (KeyError, IndexError, TypeError) as exc:
        return None, f"Unexpected response format: {exc}"


def _generate_image_gemini(
    api_key: str,
    image_model: str,
    prompt: str,
    reference_image: pathlib.Path | None,
) -> tuple[bytes | None, str | None]:
    # Gemini doesn't support image generation; fall back to OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_AI_TOKEN")
    if not openai_key:
        return None, "Gemini doesn't support image generation and no OpenAI API key available for fallback"
    print("Note: Using OpenAI for image generation since Gemini doesn't support it.")
    return _generate_image_openai(openai_key, "dall-e-3", prompt, reference_image)


def generate_image(
    provider: str,
    api_key: str,
    image_model: str,
    prompt: str,
    reference_image: pathlib.Path | None = None,
) -> tuple[bytes | None, str | None]:
    if provider == "openai":
        return _generate_image_openai(api_key, image_model, prompt, reference_image)
    return _generate_image_gemini(api_key, image_model, prompt, reference_image)


def build_colored_prompt(
    description: str,
    base_prompt: str,
    colors: list[str],
    page_index: int,
    reference_image: pathlib.Path | None,
) -> str:
    """Build the prompt for the full-color version of a page."""
    key_text = color_key_text(colors)
    prompt = (
        f"Full-color illustration for a color-by-number activity book. "
        f"Page {page_index}: {description}. "
        f"{base_prompt}. "
        f"CRITICAL COLOR RULES: Use ONLY and EXACTLY these colors: {key_text}. "
        f"Each number must ALWAYS correspond to the same color throughout the entire image. "
        f"1={colors[0]}, 2={colors[1] if len(colors)>1 else colors[0]}, etc. "
        f"Do not use any other colors, shades, tints, or color variations. "
        f"CONSISTENT COLOR MAPPING: If you use color {colors[0] if colors else 'Red'}, it must ALWAYS be number 1. "
        "LARGE SIMPLE REGIONS: Create very large, bold, simple areas of solid flat color. "
        "Each colored region should be big enough to easily fit numbers inside. "
        "THICK BLACK OUTLINES: Use bold black outlines around all shapes and objects. "
        "NO gradients, NO blending, NO subtle color variations, NO off-shades. "
        "Simple cartoon style with BIG shapes and LARGE color areas. "
        "Minimalist design with fewer details but bigger, clearer regions. "
        "COMPLETE PICTURE: all characters, objects, and scene elements are FULLY contained within the page. "
        "NO cropped subjects, NO cut-off elements at any edge. Leave comfortable white margins."
    )
    if reference_image:
        prompt += " Use the reference image for style and composition cues only."
    return prompt


def color_name_to_rgb(color_name: str) -> tuple[int, int, int]:
    """Convert a color name to an RGB tuple."""
    key = color_name.strip().lower()
    if key in COLOR_NAME_MAP:
        return COLOR_NAME_MAP[key]
    # Try PIL's ImageColor for any valid CSS color name
    try:
        from PIL import ImageColor
        rgb = ImageColor.getrgb(color_name)
        return rgb[0], rgb[1], rgb[2]
    except Exception:
        pass
    return (128, 128, 128)  # fallback gray


def _draw_number(
    draw: "ImageDraw.ImageDraw",  # type: ignore[name-defined]
    cx: int,
    cy: int,
    number: int,
    font: object,
    font_size: int,
) -> None:
    """Draw a number centered at (cx, cy) with a white halo for readability."""
    text = str(number)
    halo = max(2, font_size // 8)
    # Use textbbox for accurate centering when available (Pillow >= 8.0)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)  # type: ignore[attr-defined]
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except AttributeError:
        tw, th = font_size, font_size  # fallback estimate
    tx = cx - tw // 2
    ty = cy - th // 2
    # White halo so the digit is legible over any background
    for dx in (-halo, 0, halo):
        for dy in (-halo, 0, halo):
            if dx != 0 or dy != 0:
                draw.text((tx + dx, ty + dy), text, fill=(255, 255, 255), font=font)
    draw.text((tx, ty), text, fill=(0, 0, 0), font=font)


def _add_color_legend(
    bw_img: "Image.Image",  # type: ignore[name-defined] 
    draw: "ImageDraw.ImageDraw",  # type: ignore[name-defined]
    colors: list[str],
    font: object,
    font_size: int,
    width: int,
    height: int,
) -> None:
    """Add a color legend at the bottom of the B&W image."""
    legend_text = " | ".join(f"{i+1}={color}" for i, color in enumerate(colors))
    
    # Calculate text dimensions
    try:
        bbox = draw.textbbox((0, 0), legend_text, font=font)  # type: ignore[attr-defined]
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        text_width = len(legend_text) * font_size // 2  # fallback estimate
        text_height = font_size
    
    # Position at bottom center with some padding
    padding = 10
    legend_x = (width - text_width) // 2
    legend_y = height - text_height - padding
    
    # Draw white background rectangle for legend
    rect_padding = 5
    draw.rectangle([
        legend_x - rect_padding, 
        legend_y - rect_padding,
        legend_x + text_width + rect_padding,
        legend_y + text_height + rect_padding
    ], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
    
    # Draw the legend text
    draw.text((legend_x, legend_y), legend_text, fill=(0, 0, 0), font=font)


def create_bw_numbered_from_colored(
    colored_path: pathlib.Path,
    bw_path: pathlib.Path,
    colors: list[str],
) -> None:
    """Deterministically create a B&W numbered coloring page from a colored image.

    Algorithm:
    1. Load the colored image and strictly quantize every pixel to the nearest palette color
    2. Create clean color regions by aggressive quantization to ensure only palette colors exist
    3. Build an edge mask: pixels that border different color regions are marked as edges
    4. For each color, find all connected components and place numbers at their centroids
    5. Ensure numbers are placed well inside regions, not near edges
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from scipy.ndimage import label as scipy_label
    except ImportError as exc:
        raise SystemExit(
            "Pillow, numpy, and scipy are required for the B&W numbered conversion. "
            "Install with: pip install Pillow numpy scipy"
        ) from exc

    n_colors = len(colors)
    palette_rgb = [color_name_to_rgb(c) for c in colors]

    # Build a 256-entry PIL palette (PIL requires exactly 256 * 3 = 768 values)
    flat_palette: list[int] = []
    for r, g, b in palette_rgb:
        flat_palette.extend([r, g, b])
    flat_palette.extend([0] * (768 - len(flat_palette)))
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(flat_palette)

    # Load and aggressively quantize to ensure only palette colors exist
    img = Image.open(colored_path).convert("RGB")
    
    # First quantization pass
    quantized = img.quantize(palette=pal_img, dither=0)
    
    # Convert back to RGB and then snap each pixel to nearest palette color for extra precision
    rgb_array = np.array(quantized.convert("RGB"))
    H, W, _ = rgb_array.shape
    
    # Snap every pixel to exact palette colors
    for y in range(H):
        for x in range(W):
            pixel = rgb_array[y, x]
            # Find nearest palette color
            distances = [sum((pixel[i] - palette_rgb[c][i])**2 for i in range(3)) for c in range(len(palette_rgb))]
            nearest_color_idx = distances.index(min(distances))
            rgb_array[y, x] = palette_rgb[nearest_color_idx]
    
    # Convert to label array (0 to n_colors-1)
    labels = np.zeros((H, W), dtype=np.int32)
    for y in range(H):
        for x in range(W):
            pixel = tuple(rgb_array[y, x])
            for i, pal_color in enumerate(palette_rgb):
                if pixel == pal_color:
                    labels[y, x] = i
                    break

    # Edge mask: True where a pixel borders a pixel of a different label
    edge_mask = np.zeros((H, W), dtype=bool)
    edge_mask[:-1, :] |= labels[:-1, :] != labels[1:, :]   # pixel vs pixel below
    edge_mask[1:, :] |= labels[:-1, :] != labels[1:, :]    # mirror upward
    edge_mask[:, :-1] |= labels[:, :-1] != labels[:, 1:]   # pixel vs pixel right
    edge_mask[:, 1:] |= labels[:, :-1] != labels[:, 1:]    # mirror leftward

    # Expand edge mask by a few pixels to ensure numbers are well inside regions
    for _ in range(3):  # Dilate edge mask 3 times
        new_edge_mask = edge_mask.copy()
        new_edge_mask[:-1, :] |= edge_mask[1:, :]   # down
        new_edge_mask[1:, :] |= edge_mask[:-1, :]   # up  
        new_edge_mask[:, :-1] |= edge_mask[:, 1:]   # right
        new_edge_mask[:, 1:] |= edge_mask[:, :-1]   # left
        edge_mask = new_edge_mask

    # White canvas; paint edges black
    bw_array = np.full((H, W, 3), 255, dtype=np.uint8)
    bw_array[edge_mask] = 0
    bw_img = Image.fromarray(bw_array)
    draw = ImageDraw.Draw(bw_img)

    # Choose a proportional font size
    font_size = max(16, min(W // 40, H // 40))  # Slightly bigger fonts
    font: object
    for font_path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except OSError:
            continue
    else:
        font = ImageFont.load_default()

    # For each color that exists in the image, find connected components and place numbers
    placed_numbers = 0
    for color_idx in range(n_colors):
        # Create mask for this color only
        color_mask = (labels == color_idx) & ~edge_mask
        
        if not np.any(color_mask):
            continue  # This color doesn't appear in non-edge regions
            
        # Find connected components for this color
        labeled_regions, num_regions = scipy_label(color_mask)
        
        # For each connected component of this color, place a number at its centroid
        for region_id in range(1, num_regions + 1):
            region_mask = labeled_regions == region_id
            region_coords = np.argwhere(region_mask)
            
            if len(region_coords) < 50:  # Skip very small regions
                continue
                
            # Calculate centroid
            cy = int(np.mean(region_coords[:, 0]))
            cx = int(np.mean(region_coords[:, 1]))
            
            # Double-check this position is safe (not on edge and correct color)
            if not edge_mask[cy, cx] and labels[cy, cx] == color_idx:
                _draw_number(draw, cx, cy, color_idx + 1, font, font_size)
                placed_numbers += 1
                
    # Fallback: if very few numbers were placed, use centroid approach for major regions
    if placed_numbers < len(colors) // 2:
        for color_idx in range(n_colors):
            color_only_mask = (labels == color_idx)
            if not np.any(color_only_mask):
                continue
                
            # Find largest connected component for this color (ignoring edge constraints)
            labeled_regions, num_regions = scipy_label(color_only_mask)
            if num_regions == 0:
                continue
                
            # Find the largest region
            largest_region_id = 0
            largest_region_size = 0
            for region_id in range(1, num_regions + 1):
                region_size = np.sum(labeled_regions == region_id)
                if region_size > largest_region_size:
                    largest_region_size = region_size
                    largest_region_id = region_id
                    
            if largest_region_id > 0:
                region_coords = np.argwhere(labeled_regions == largest_region_id)
                cy = int(np.mean(region_coords[:, 0]))
                cx = int(np.mean(region_coords[:, 1]))
                _draw_number(draw, cx, cy, color_idx + 1, font, font_size)

    # Add color legend at the bottom of the image
    _add_color_legend(bw_img, draw, colors, font, font_size, W, H)
    
    bw_img.save(str(bw_path), "JPEG")


# ---------------------------------------------------------------------------
# Single-page generation (called from thread pool)
# ---------------------------------------------------------------------------

def generate_single_page(
    page_info: tuple,
) -> tuple[int, pathlib.Path | None, pathlib.Path | None, dict | None]:
    """Generate one page (colored + deterministic B&W).

    Returns (page_index, colored_path, bw_path, failure_info).
    The B&W numbered version is derived deterministically from the colored image
    via palette quantization + edge detection, not by a second AI generation call.
    """
    (
        i,
        description,
        base_prompt,
        colors,
        provider,
        api_key,
        image_model,
        base_name,
        reference_image,
        output_dir,
        page_num_str,
    ) = page_info

    colored_file = output_dir / f"{base_name}-page-{page_num_str}-colored.jpg"
    bw_file = output_dir / f"{base_name}-page-{page_num_str}-bw.jpg"

    # --- Step 1: generate the colored version with AI ---
    colored_prompt = build_colored_prompt(description, base_prompt, colors, i, reference_image)
    print(f"Generating page {i} (colored): {description[:80]}{'...' if len(description) > 80 else ''}")
    colored_bytes, colored_err = generate_image(provider, api_key, image_model, colored_prompt, reference_image)

    if colored_bytes is None:
        failure_info = {
            "page": i,
            "description": description,
            "error": f"colored: {colored_err}",
        }
        print(f"⚠️  Page {i} failed: {failure_info['error']}")
        return i, None, None, failure_info

    convert_to_jpg(colored_bytes, colored_file)
    print(f"✅ Page {i} colored: {colored_file.name}")

    # --- Step 2: derive the B&W numbered version deterministically ---
    print(f"Creating page {i} (B&W numbered from colored image)...")
    try:
        create_bw_numbered_from_colored(colored_file, bw_file, colors)
        print(f"✅ Page {i} B&W: {bw_file.name}")
    except Exception as exc:
        failure_info = {
            "page": i,
            "description": description,
            "error": f"b&w conversion failed: {exc}",
        }
        print(f"⚠️  Page {i} B&W conversion failed: {exc}")
        # Return colored as partial success; bw is None to signal the failure
        return i, colored_file, None, failure_info

    return i, colored_file, bw_file, None


def convert_to_jpg(png_bytes: bytes, output_path: pathlib.Path) -> None:
    """Convert PNG bytes to a JPEG file using sips."""
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        tmp.write(png_bytes)
        tmp.flush()
        result = subprocess.run(
            ["sips", "-s", "format", "jpeg", tmp.name, "--out", str(output_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(f"sips conversion failed: {result.stderr}")


# ---------------------------------------------------------------------------
# Replay from markdown
# ---------------------------------------------------------------------------

def load_plan_from_markdown(md_path: pathlib.Path) -> tuple[
    dict, str, list[str], int, str | None, str | None, str | None, str, str, pathlib.Path | None
]:
    """Parse a previously generated plan Markdown file for replay."""
    content = md_path.read_text()

    theme = ""
    colors: list[str] = []
    pages = DEFAULT_PAGES
    provider = None
    planner_model = None
    image_model = None
    reference_image = None
    planner_system_prompt = ""
    planner_user_prompt = ""

    if m := re.search(r"\*{0,2}Theme\*{0,2}:\s*(.+)", content):
        theme = m.group(1).strip()
    if m := re.search(r"\*{0,2}Colors\*{0,2}:\s*(.+)", content):
        colors = [c.strip() for c in m.group(1).split(",") if c.strip()]
    if m := re.search(r"\*{0,2}Number of pages\*{0,2}:\s*(\d+)", content):
        pages = int(m.group(1))
    if m := re.search(r"\*{0,2}Provider\*{0,2}:\s*(.+)", content):
        provider = m.group(1).strip()
    if m := re.search(r"\*{0,2}Planner model\*{0,2}:\s*(.+)", content):
        planner_model = m.group(1).strip()
    if m := re.search(r"\*{0,2}Image model\*{0,2}:\s*(.+)", content):
        image_model = m.group(1).strip()
    if m := re.search(r"\*{0,2}Reference image\*{0,2}:\s*(.+)", content):
        ref_path = m.group(1).strip()
        if ref_path not in ("None", "-"):
            reference_image = pathlib.Path(ref_path)

    picture_descriptions: list[str] = []
    desc_section = re.search(r"## Picture Descriptions\s*(.+?)(?=##|$)", content, re.DOTALL)
    if desc_section:
        for line in desc_section.group(1).strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                picture_descriptions.append(line[2:])

    if not colors:
        colors = list(DEFAULT_COLORS)

    key_text = color_key_text(colors)
    plan = {
        "theme_title": theme[:80].strip(),
        "picture_descriptions": picture_descriptions,
        "base_prompt": (
            f"Simple flat graphic style color-by-number activity illustration. Color palette: {key_text}."
        ),
    }

    return (
        plan,
        theme,
        colors,
        pages,
        provider,
        planner_model,
        image_model,
        planner_system_prompt,
        planner_user_prompt,
        reference_image,
    )


# ---------------------------------------------------------------------------
# Markdown plan log
# ---------------------------------------------------------------------------

def write_plan_log(
    log_path: pathlib.Path,
    provider: str,
    theme: str,
    colors: list[str],
    pages: int,
    reference_image: pathlib.Path | None,
    planner_model: str,
    image_model: str,
    plan: dict,
    planner_system_prompt: str,
    planner_user_prompt: str,
    colored_files: list[pathlib.Path],
    bw_files: list[pathlib.Path],
    failed_pages: list[dict],
    replay_source: pathlib.Path | None,
) -> None:
    reference_text = str(reference_image) if reference_image else "None"
    colors_text = ", ".join(colors)

    files_section = ""
    if colored_files:
        files_section += "## Successfully Generated Files\n\n"
        pairs = list(zip(colored_files, bw_files))
        for colored, bw in pairs:
            files_section += f"- {colored.name} *(colored)*\n"
            files_section += f"- {bw.name} *(B&W numbered)*\n"
        files_section += "\n"

    if failed_pages:
        files_section += "## Failed Pages\n\n"
        for failure in failed_pages:
            files_section += f"- **Page {failure['page']}**: {failure['error']}\n"
            files_section += f"  - Description: {failure['description']}\n"
        files_section += "\n"

    content = f"""# Color-by-Number Generation Plan

Generated: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{"Replayed from: " + str(replay_source) if replay_source else "Original generation"}

## Inputs

- **Theme**: {theme}
- **Colors**: {colors_text}
- **Number of pages**: {pages}
- **Reference image**: {reference_text}
- **Provider**: {provider}
- **Planner model**: {planner_model}
- **Image model**: {image_model}

## Color Key

{color_key_markdown(colors)}

## Main Characters

{chr(10).join(f"- {char}" for char in plan.get("main_characters", [])) or "- Not specified"}

## Picture Descriptions

{chr(10).join(f"- {desc}" for desc in plan["picture_descriptions"])}

## Base Style Prompt

```text
{plan["base_prompt"]}
```

{files_section}## Planner System Prompt

```text
{planner_system_prompt}
```

## Planner User Prompt

```json
{planner_user_prompt}
```
"""
    log_path.write_text(content)


# ---------------------------------------------------------------------------
# Tooling check
# ---------------------------------------------------------------------------

def ensure_tooling() -> None:
    import shutil
    for tool in ("sips",):
        if not shutil.which(tool):
            raise SystemExit(f"Required tool not found in PATH: {tool}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    ensure_tooling()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    replay_source: pathlib.Path | None = None
    planner_system_prompt = ""
    planner_user_prompt = ""
    reference_image: pathlib.Path | None = resolve_reference_image_path(args.reference_image)

    if args.replay_from_md:
        replay_source = pathlib.Path(args.replay_from_md).expanduser().resolve()
        if not replay_source.is_file():
            raise SystemExit(f"Replay Markdown file not found: {replay_source}")
        (
            plan,
            theme,
            colors,
            pages,
            replay_provider,
            replay_planner_model,
            replay_image_model,
            planner_system_prompt,
            planner_user_prompt,
            replay_reference_image,
        ) = load_plan_from_markdown(replay_source)
        provider = replay_provider or args.provider or default_provider()
        planner_model = (
            args.planner_model
            or replay_planner_model
            or default_planner_model(provider, get_api_key(provider) if provider == "gemini" else None)
        )
        image_model = args.image_model or replay_image_model or default_image_model(provider)
        if reference_image is None:
            reference_image = replay_reference_image
        # Allow CLI --colors to override colors from the markdown
        if args.colors:
            colors = parse_colors(args.colors)
    else:
        if args.theme is None:
            raise SystemExit("Provide --theme or use --replay-from-md.")
        provider = args.provider or default_provider()
        theme = args.theme
        colors = parse_colors(args.colors)
        pages = args.pages
        planner_model = (
            args.planner_model
            or default_planner_model(provider, get_api_key(provider) if provider == "gemini" else None)
        )
        image_model = args.image_model or default_image_model(provider)
        api_key = get_api_key(provider)
        plan, planner_system_prompt, planner_user_prompt = build_plan(
            provider=provider,
            api_key=api_key,
            planner_model=planner_model,
            theme=theme,
            colors=colors,
            pages=pages,
        )

    if args.replay_from_md:
        api_key = get_api_key(provider)

    if pages < 1 or pages > 20:
        raise SystemExit("--pages must be between 1 and 20.")

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"{slugify(theme)}-{timestamp}"
    log_path = output_dir / f"{base_name}-plan.md"

    colored_files: list[pathlib.Path] = []
    bw_files: list[pathlib.Path] = []
    failed_pages: list[dict] = []

    descriptions = plan["picture_descriptions"]

    # Build page info tuples for all pages
    page_info_list = []
    for i, description in enumerate(descriptions, 1):
        page_num_str = f"{i:03d}"
        page_info = (
            i,
            description,
            plan["base_prompt"],
            colors,
            provider,
            api_key,
            image_model,
            base_name,
            reference_image,
            output_dir,
            page_num_str,
        )
        page_info_list.append(page_info)

    print(f"Generating {pages} pages (2 images each) in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(generate_single_page, page_info) for page_info in page_info_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                page_idx, colored_file, bw_file, failure_info = future.result()
                if colored_file is not None and bw_file is not None:
                    colored_files.append(colored_file)
                    bw_files.append(bw_file)
                else:
                    failed_pages.append(failure_info)
            except Exception as exc:
                print(f"⚠️  Page generation error: {exc}")
                failed_pages.append({
                    "page": "unknown",
                    "description": "unknown",
                    "error": f"Parallel generation error: {exc}",
                })

    # Sort output files by page number for consistent ordering
    colored_files.sort(key=lambda p: p.name)
    bw_files.sort(key=lambda p: p.name)

    write_plan_log(
        log_path=log_path,
        provider=provider,
        theme=theme,
        colors=colors,
        pages=pages,
        reference_image=reference_image,
        planner_model=planner_model,
        image_model=image_model,
        plan=plan,
        planner_system_prompt=planner_system_prompt,
        planner_user_prompt=planner_user_prompt,
        colored_files=colored_files,
        bw_files=bw_files,
        failed_pages=failed_pages,
        replay_source=replay_source,
    )

    print(f"📝 Created {log_path}")

    successful_count = len(colored_files)
    failed_count = len(failed_pages)

    if successful_count > 0:
        print(f"✅ Generated {successful_count} color-by-number page pairs successfully")
        print(f"   Color key: {color_key_text(colors)}")

    if failed_count > 0:
        print(f"❌ {failed_count} pages failed to generate:")
        for failure in failed_pages:
            print(f"   • Page {failure['page']}: {failure['error']}")
        print("\n💡 Tip: Try rephrasing problematic descriptions to avoid content filters")

    return 0 if successful_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
