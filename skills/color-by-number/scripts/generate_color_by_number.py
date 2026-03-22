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
DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_PLANNER_MODEL = "gpt-4o"
DEFAULT_OPENAI_IMAGE_MODEL = "dall-e-3"
DEFAULT_GEMINI_PLANNER_MODEL = "gemini-pro"
DEFAULT_GEMINI_IMAGE_MODEL = "dall-e-3"  # Fallback to OpenAI for image generation
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
                "describe objects in terms of palette colors"
            ),
            "framing": (
                "ensure all characters, objects, and scene elements are COMPLETELY contained within page boundaries "
                "with comfortable white margins; NO cropped or cut-off elements at any edge"
            ),
            "complexity": "appropriate detail level for a color-by-number activity for children",
        },
        "output_schema": {
            "theme_title": "short human-readable title for the story",
            "main_characters": [
                "array of 1-3 main characters with detailed physical descriptions for consistency"
            ],
            "picture_descriptions": [
                f"array of exactly {pages} distinct picture concepts that advance the story with the same characters"
            ],
            "base_prompt": (
                "one detailed style prompt used for all images, referencing the palette colors and "
                "simple flat graphic design suitable for a color-by-number activity"
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
    if not isinstance(picture_descriptions, list) or not all(
        isinstance(item, str) and item.strip() for item in picture_descriptions
    ):
        raise SystemExit(f"Planner returned invalid picture descriptions:\n{json.dumps(plan, indent=2)}")
    if len(picture_descriptions) != pages:
        raise SystemExit(
            f"Planner returned {len(picture_descriptions)} pictures; expected {pages}."
        )

    main_characters = plan.get("main_characters", [])
    if not isinstance(main_characters, list):
        main_characters = []

    base_prompt = plan.get("base_prompt")
    if not isinstance(base_prompt, str) or not base_prompt.strip():
        char_desc = " Main characters: " + "; ".join(main_characters) if main_characters else ""
        key_text = color_key_text(colors)
        plan["base_prompt"] = (
            f"Simple flat graphic style color-by-number activity illustration. "
            f"Color palette: {key_text}.{char_desc}"
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
        f"Use ONLY these colors: {key_text}. "
        "Bright, vivid, flat graphic style with clear distinct color regions. "
        "Simple design suitable for children. "
        "COMPLETE PICTURE: all characters, objects, and scene elements are FULLY contained within the page. "
        "NO cropped subjects, NO cut-off elements at any edge. Leave comfortable white margins."
    )
    if reference_image:
        prompt += " Use the reference image for style and composition cues only."
    return prompt


def build_bw_prompt(
    description: str,
    base_prompt: str,
    colors: list[str],
    page_index: int,
    reference_image: pathlib.Path | None,
) -> str:
    """Build the prompt for the black-and-white numbered version of a page."""
    key_text = color_key_text(colors)
    num_colors = len(colors)
    prompt = (
        f"Black and white color-by-number activity page for children. "
        f"Page {page_index}: {description}. "
        f"{base_prompt}. "
        "STYLE: clean thick black outlines on pure white background, flat simplified shapes, no shading, no gradients. "
        f"NUMBER REGIONS: place a small bold number (1 through {num_colors}) inside each distinct color region "
        f"according to this color key: {key_text}. "
        "The numbers must be clearly legible inside their regions. "
        "COMPLETE PICTURE: all characters, objects, and scene elements are FULLY contained within the page. "
        "NO cropped subjects, NO cut-off elements at any edge. Leave comfortable white margins."
    )
    if reference_image:
        prompt += " Use the reference image for composition and layout cues only."
    return prompt


# ---------------------------------------------------------------------------
# Single-page generation (called from thread pool)
# ---------------------------------------------------------------------------

def generate_single_page(
    page_info: tuple,
) -> tuple[int, pathlib.Path | None, pathlib.Path | None, dict | None]:
    """Generate one page (colored + B&W). Returns (page_index, colored_path, bw_path, failure_info)."""
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

    colored_prompt = build_colored_prompt(description, base_prompt, colors, i, reference_image)
    bw_prompt = build_bw_prompt(description, base_prompt, colors, i, reference_image)

    print(f"Generating page {i} (colored): {description[:80]}{'...' if len(description) > 80 else ''}")
    colored_bytes, colored_err = generate_image(provider, api_key, image_model, colored_prompt, reference_image)

    print(f"Generating page {i} (B&W numbered): {description[:80]}{'...' if len(description) > 80 else ''}")
    bw_bytes, bw_err = generate_image(provider, api_key, image_model, bw_prompt, None)

    # If either generation failed, record a failure
    if colored_bytes is None or bw_bytes is None:
        errors = []
        if colored_bytes is None:
            errors.append(f"colored: {colored_err}")
        if bw_bytes is None:
            errors.append(f"b&w: {bw_err}")
        failure_info = {
            "page": i,
            "description": description,
            "error": "; ".join(errors),
        }
        print(f"⚠️  Page {i} failed: {failure_info['error']}")
        return i, None, None, failure_info

    convert_to_jpg(colored_bytes, colored_file)
    convert_to_jpg(bw_bytes, bw_file)
    print(f"✅ Page {i}: {colored_file.name} + {bw_file.name}")
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
