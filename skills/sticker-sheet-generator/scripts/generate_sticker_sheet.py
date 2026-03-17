#!/usr/bin/env python3
"""Generate a printable A4 sticker sheet from theme/style prompts."""

from __future__ import annotations

import argparse
import base64
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
IMAGE_EDITS_URL = "https://api.openai.com/v1/images/edits"
FILES_URL = "https://api.openai.com/v1/files"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_PLANNER_MODEL = "gpt-4.1"
DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-1.5"
DEFAULT_GEMINI_PLANNER_MODEL = "gemini-2.5-pro"
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-3-pro-image-preview"
FOOTER_TEXT = "Nachala, the one and only!"
A4_WIDTH = 2480
A4_HEIGHT = 3508


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an A4 sticker sheet PNG, JPG, and prompt log.",
    )
    parser.add_argument("--theme", help="Theme prompt or description.")
    parser.add_argument("--style", help="Style prompt or description.")
    parser.add_argument(
        "--stickers-per-page",
        type=int,
        help="Exact number of stickers to place on the page.",
    )
    parser.add_argument(
        "--replay-from-md",
        help="Reuse an existing prompts Markdown file as editable input.",
    )
    parser.add_argument(
        "--style-reference-image",
        help="Optional local image path to use as a style reference during generation.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "gemini"),
        help="API provider to use for planning and image generation. If omitted, Gemini is preferred when a Gemini key is available; otherwise OpenAI is used.",
    )
    parser.add_argument(
        "--output-dir",
        default="tmp",
        help="Directory for output files. Defaults to ./tmp",
    )
    parser.add_argument(
        "--planner-model",
        help="Text model used to plan sticker descriptions. Provider-specific default is used if omitted.",
    )
    parser.add_argument(
        "--image-model",
        help="Image model used to render the sheet. Provider-specific default is used if omitted.",
    )
    return parser.parse_args()


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
    if has_gemini_key():
        return "gemini"
    return DEFAULT_PROVIDER


def default_planner_model(provider: str) -> str:
    if provider == "gemini":
        return DEFAULT_GEMINI_PLANNER_MODEL
    return DEFAULT_OPENAI_PLANNER_MODEL


def default_image_model(provider: str) -> str:
    if provider == "gemini":
        return DEFAULT_GEMINI_IMAGE_MODEL
    return DEFAULT_OPENAI_IMAGE_MODEL


def resolve_reference_image_path(image_path: str | None) -> pathlib.Path | None:
    if not image_path:
        return None
    if image_path.strip() == "-":
        return None
    path = pathlib.Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Style reference image not found: {path}")
    return path


def guess_mime_type(image_path: pathlib.Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type or not mime_type.startswith("image/"):
        raise SystemExit(f"Unsupported or unknown image type for style reference: {image_path}")
    return mime_type


def infer_provider_from_model(model: str | None) -> str | None:
    if not model:
        return None
    normalized = model.strip().lower()
    if normalized.startswith("gemini-"):
        return "gemini"
    if normalized.startswith("gpt-") or normalized.startswith("o"):
        return "openai"
    return None


def resolve_replay_provider(
    explicit_provider: str | None,
    replay_provider: str | None,
    replay_planner_model: str | None,
    replay_image_model: str | None,
) -> str:
    if explicit_provider:
        return explicit_provider
    inferred = (
        infer_provider_from_model(replay_image_model)
        or infer_provider_from_model(replay_planner_model)
    )
    if inferred:
        return inferred
    if replay_provider in {"openai", "gemini"}:
        return replay_provider
    return default_provider()


def resolve_replay_model(
    provider: str,
    replay_model: str | None,
    cli_model: str | None,
    default_model: str,
) -> str:
    if cli_model:
        return cli_model
    if replay_model and infer_provider_from_model(replay_model) in {None, provider}:
        return replay_model
    return default_model


def slugify(value: str) -> str:
    slug = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return (slug[:48].rstrip("-")) or "sticker-sheet"


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
        raise SystemExit(f"OpenAI API request failed: HTTP {exc.code}\n{error_body}") from exc
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


def post_openai_multipart(
    url: str,
    fields: list[tuple[str, str]],
    files: list[tuple[str, pathlib.Path, str]],
    api_key: str,
) -> dict:
    boundary = f"----CodexBoundary{dt.datetime.now().timestamp():.6f}".replace(".", "")
    body = bytearray()

    for name, value in fields:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8")
        )

    for field_name, file_path, mime_type in files:
        filename = file_path.name
        data = file_path.read_bytes()
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
                f"Content-Type: {mime_type}\r\n\r\n"
            ).encode("utf-8")
        )
        body.extend(data)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    request = urllib.request.Request(
        url,
        data=bytes(body),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"OpenAI multipart request failed: HTTP {exc.code}\n{error_body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Network request failed: {exc.reason}") from exc


def upload_openai_file(api_key: str, image_path: pathlib.Path) -> str:
    response = post_openai_multipart(
        FILES_URL,
        fields=[("purpose", "vision")],
        files=[("file", image_path, guess_mime_type(image_path))],
        api_key=api_key,
    )
    file_id = response.get("id")
    if not isinstance(file_id, str) or not file_id:
        raise SystemExit(f"Unexpected OpenAI file upload response: {json.dumps(response, indent=2)}")
    return file_id


def planner_messages(theme: str, style: str, stickers_per_page: int) -> tuple[str, str]:
    system_prompt = (
        "You are a sticker-sheet planner. Return valid JSON only. "
        "Create exactly the requested number of distinct sticker descriptions and one production-ready "
        "image prompt for a portrait sticker sheet. The sheet must reserve empty transparent space at the bottom "
        "for a footer that will be added later. Do not include markdown fences."
    )
    user_payload = {
        "task": "Plan a printable die-cut sticker sheet",
        "requirements": {
            "theme": theme,
            "style": style,
            "stickers_per_page": stickers_per_page,
            "page_orientation": "A4 portrait",
            "background": "transparent",
            "cut_contour": "every sticker must have a bold black outer contour suitable for cutting",
            "layout": "stickers separated, non-overlapping, not touching edges",
            "footer_safe_area": (
                "leave a clean empty transparent area at the bottom for the exact footer text "
                f"'{FOOTER_TEXT}' to be added outside the model later"
            ),
            "text_policy": "avoid visible text inside stickers unless tiny incidental decoration",
        },
        "output_schema": {
            "theme_title": "short human-readable title",
            "sticker_descriptions": [
                "array of exact length stickers_per_page with distinct sticker concepts"
            ],
            "image_prompt": (
                "one detailed prompt that includes the sticker list, layout rules, contour rules, "
                "transparent background, footer safe area, and strong production wording"
            ),
        },
    }
    return system_prompt, json.dumps(user_payload, ensure_ascii=True)


def build_sticker_plan_openai(
    api_key: str,
    planner_model: str,
    theme: str,
    style: str,
    stickers_per_page: int,
) -> tuple[dict, str, str]:
    system_prompt, user_prompt = planner_messages(theme, style, stickers_per_page)
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

    sticker_descriptions = plan.get("sticker_descriptions")
    if not isinstance(sticker_descriptions, list) or not all(
        isinstance(item, str) and item.strip() for item in sticker_descriptions
    ):
        raise SystemExit(f"Planner returned invalid sticker descriptions:\n{json.dumps(plan, indent=2)}")
    if len(sticker_descriptions) != stickers_per_page:
        raise SystemExit(
            f"Planner returned {len(sticker_descriptions)} stickers; expected {stickers_per_page}."
        )

    image_prompt = plan.get("image_prompt")
    if not isinstance(image_prompt, str) or not image_prompt.strip():
        image_prompt = synthesize_image_prompt(theme, style, sticker_descriptions)
        plan["image_prompt"] = image_prompt

    theme_title = plan.get("theme_title")
    if not isinstance(theme_title, str) or not theme_title.strip():
        plan["theme_title"] = theme[:80].strip()

    return plan, system_prompt, user_prompt


def build_sticker_plan_gemini(
    api_key: str,
    planner_model: str,
    theme: str,
    style: str,
    stickers_per_page: int,
) -> tuple[dict, str, str]:
    system_prompt, user_prompt = planner_messages(theme, style, stickers_per_page)
    schema = {
        "type": "object",
        "properties": {
            "theme_title": {"type": "string"},
            "sticker_descriptions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": stickers_per_page,
                "maxItems": stickers_per_page,
            },
            "image_prompt": {"type": "string"},
        },
        "required": ["theme_title", "sticker_descriptions", "image_prompt"],
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
            "responseJsonSchema": schema,
        },
    }
    response = post_gemini_json(planner_model, payload, api_key)

    parts = (
        response.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    text_parts = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")]
    content = "\n".join(text_parts).strip()
    if not content:
        raise SystemExit(f"Unexpected Gemini planner response: {json.dumps(response, indent=2)}")

    try:
        plan = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Gemini planner did not return valid JSON:\n{content}") from exc

    sticker_descriptions = plan.get("sticker_descriptions")
    if not isinstance(sticker_descriptions, list) or not all(
        isinstance(item, str) and item.strip() for item in sticker_descriptions
    ):
        raise SystemExit(
            f"Gemini planner returned invalid sticker descriptions:\n{json.dumps(plan, indent=2)}"
        )
    if len(sticker_descriptions) != stickers_per_page:
        raise SystemExit(
            f"Gemini planner returned {len(sticker_descriptions)} stickers; expected {stickers_per_page}."
        )

    image_prompt = plan.get("image_prompt")
    if not isinstance(image_prompt, str) or not image_prompt.strip():
        image_prompt = synthesize_image_prompt(theme, style, sticker_descriptions)
        plan["image_prompt"] = image_prompt

    theme_title = plan.get("theme_title")
    if not isinstance(theme_title, str) or not theme_title.strip():
        plan["theme_title"] = theme[:80].strip()

    return plan, system_prompt, user_prompt


def build_sticker_plan(
    provider: str,
    api_key: str,
    planner_model: str,
    theme: str,
    style: str,
    stickers_per_page: int,
) -> tuple[dict, str, str]:
    if provider == "gemini":
        return build_sticker_plan_gemini(
            api_key=api_key,
            planner_model=planner_model,
            theme=theme,
            style=style,
            stickers_per_page=stickers_per_page,
        )
    return build_sticker_plan_openai(
        api_key=api_key,
        planner_model=planner_model,
        theme=theme,
        style=style,
        stickers_per_page=stickers_per_page,
    )


def synthesize_image_prompt(theme: str, style: str, sticker_descriptions: list[str]) -> str:
    bullet_list = "\n".join(f"- {item}" for item in sticker_descriptions)
    return f"""Create a premium printable portrait A4 sticker sheet.

Theme input:
{theme}

Style input:
{style}

Sticker subjects:
{bullet_list}

Production requirements:
- exactly {len(sticker_descriptions)} separate stickers, all fully visible
- transparent background with empty space between stickers
- portrait layout suitable for A4 output
- bold black outer contour around every sticker for cutting
- thin white inner border inside the black contour for easy peeling
- generous spacing, no overlap, no touching page edges
- reserve a clean transparent footer safe area at the bottom of the page
- no full-page scene background
- polished, cohesive, commercially printable sticker-sheet look"""


def augment_prompt_with_style_reference(image_prompt: str) -> str:
    return (
        f"{image_prompt}\n\n"
        "Use the attached reference image primarily as a style reference for rendering approach, "
        "materials, line quality, color relationships, and overall visual language. "
        "Do not copy the exact subject matter unless the written prompt explicitly asks for it."
    )


def extract_bullet_value(markdown_text: str, label: str) -> str | None:
    pattern = re.compile(rf"^- {re.escape(label)}:\s*(.+)$", re.MULTILINE)
    match = pattern.search(markdown_text)
    if not match:
        return None
    return match.group(1).strip()


def extract_fenced_block(markdown_text: str, heading: str) -> str | None:
    pattern = re.compile(
        rf"^## {re.escape(heading)}\n\n```[^\n]*\n(.*?)\n```",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(markdown_text)
    if not match:
        return None
    return match.group(1).strip()


def extract_sticker_descriptions(markdown_text: str) -> list[str]:
    section_pattern = re.compile(
        r"^## Sticker Descriptions\n\n(.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = section_pattern.search(markdown_text)
    if not match:
        return []
    descriptions: list[str] = []
    for line in match.group(1).splitlines():
        item = re.sub(r"^\s*\d+\.\s*", "", line).strip()
        if item:
            descriptions.append(item)
    return descriptions


def load_plan_from_markdown(
    markdown_path: pathlib.Path,
) -> tuple[dict, str, str, int, str, str, str, str, str, pathlib.Path | None]:
    markdown_text = markdown_path.read_text()
    provider = extract_bullet_value(markdown_text, "Provider")
    theme = extract_bullet_value(markdown_text, "Theme")
    style = extract_bullet_value(markdown_text, "Style")
    style_reference_image = extract_bullet_value(markdown_text, "Style reference image")
    stickers_value = extract_bullet_value(markdown_text, "Stickers per page")
    planner_model = extract_bullet_value(markdown_text, "Planner model")
    image_model = extract_bullet_value(markdown_text, "Image model")
    image_prompt = extract_fenced_block(markdown_text, "Generated Image Prompt")
    planner_system_prompt = extract_fenced_block(markdown_text, "Planner System Prompt") or ""
    planner_user_prompt = extract_fenced_block(markdown_text, "Planner User Prompt") or ""
    sticker_descriptions = extract_sticker_descriptions(markdown_text)

    if not theme or not style or not stickers_value:
        raise SystemExit(
            f"Replay Markdown is missing one of Theme, Style, or Stickers per page: {markdown_path}"
        )

    try:
        stickers_per_page = int(stickers_value)
    except ValueError as exc:
        raise SystemExit(f"Invalid Stickers per page value in {markdown_path}: {stickers_value}") from exc

    if not sticker_descriptions:
        raise SystemExit(f"Replay Markdown is missing sticker descriptions: {markdown_path}")

    if len(sticker_descriptions) != stickers_per_page:
        raise SystemExit(
            f"Replay Markdown has {len(sticker_descriptions)} sticker descriptions but "
            f"Stickers per page is {stickers_per_page}."
        )

    if not image_prompt:
        image_prompt = synthesize_image_prompt(theme, style, sticker_descriptions)

    plan = {
        "theme_title": theme[:80].strip(),
        "sticker_descriptions": sticker_descriptions,
        "image_prompt": image_prompt,
    }
    return (
        plan,
        theme,
        style,
        stickers_per_page,
        provider or default_provider(),
        planner_model or default_planner_model(provider or default_provider()),
        image_model or default_image_model(provider or default_provider()),
        planner_system_prompt,
        planner_user_prompt,
        resolve_reference_image_path(style_reference_image),
    )


def generate_image_openai(
    api_key: str,
    image_model: str,
    image_prompt: str,
    style_reference_image: pathlib.Path | None = None,
) -> bytes:
    if style_reference_image is not None:
        file_id = upload_openai_file(api_key, style_reference_image)
        payload = {
            "model": image_model,
            "prompt": augment_prompt_with_style_reference(image_prompt),
            "images": [{"file_id": file_id}],
            "input_fidelity": "high",
            "background": "transparent",
            "output_format": "png",
            "quality": "high",
            "size": "1024x1536",
        }
        response = post_json(IMAGE_EDITS_URL, payload, api_key)
        try:
            image_b64 = response["data"][0]["b64_json"]
        except (KeyError, IndexError, TypeError) as exc:
            raise SystemExit(f"Unexpected image edit response: {json.dumps(response, indent=2)}") from exc
        return base64.b64decode(image_b64)

    payload = {
        "model": image_model,
        "prompt": image_prompt,
        "background": "transparent",
        "output_format": "png",
        "quality": "high",
        "size": "1024x1536",
    }
    response = post_json(IMAGES_URL, payload, api_key)
    try:
        image_b64 = response["data"][0]["b64_json"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SystemExit(f"Unexpected image response: {json.dumps(response, indent=2)}") from exc
    return base64.b64decode(image_b64)


def generate_image_gemini(
    api_key: str,
    image_model: str,
    image_prompt: str,
    style_reference_image: pathlib.Path | None = None,
) -> bytes:
    parts: list[dict] = [{"text": image_prompt}]
    if style_reference_image is not None:
        parts.append(
            {
                "inlineData": {
                    "mimeType": guess_mime_type(style_reference_image),
                    "data": base64.b64encode(style_reference_image.read_bytes()).decode("utf-8"),
                }
            }
        )
        parts[0] = {"text": augment_prompt_with_style_reference(image_prompt)}
    payload = {"contents": [{"parts": parts}]}
    response = post_gemini_json(image_model, payload, api_key)
    candidates = response.get("candidates", [])
    for candidate in candidates:
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            inline_data = part.get("inlineData")
            if not inline_data:
                continue
            data = inline_data.get("data")
            if data:
                return base64.b64decode(data)
    raise SystemExit(f"Unexpected Gemini image response: {json.dumps(response, indent=2)}")


def generate_image(
    provider: str,
    api_key: str,
    image_model: str,
    image_prompt: str,
    style_reference_image: pathlib.Path | None = None,
) -> bytes:
    if provider == "gemini":
        return generate_image_gemini(api_key, image_model, image_prompt, style_reference_image)
    return generate_image_openai(api_key, image_model, image_prompt, style_reference_image)


def render_a4_png(raw_png_bytes: bytes, output_png_path: pathlib.Path, footer_text: str) -> None:
    swift_script = pathlib.Path(__file__).with_name("render_a4_with_footer.swift")
    with tempfile.TemporaryDirectory(prefix="sticker-sheet-") as temp_dir:
        raw_path = pathlib.Path(temp_dir) / "raw.png"
        module_cache_path = pathlib.Path(temp_dir) / "swift-module-cache"
        module_cache_path.mkdir()
        raw_path.write_bytes(raw_png_bytes)
        result = subprocess.run(
            [
                "swift",
                "-module-cache-path",
                str(module_cache_path),
                str(swift_script),
                str(raw_path),
                str(output_png_path),
                footer_text,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise SystemExit(
            "Failed to render the A4 PNG.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def export_jpg(png_path: pathlib.Path, jpg_path: pathlib.Path) -> None:
    result = subprocess.run(
        ["sips", "-s", "format", "jpeg", str(png_path), "--out", str(jpg_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            "PNG was created, but JPG export failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def write_prompt_log(
    log_path: pathlib.Path,
    provider: str,
    theme: str,
    style: str,
    style_reference_image: pathlib.Path | None,
    stickers_per_page: int,
    planner_model: str,
    image_model: str,
    plan: dict,
    planner_system_prompt: str,
    planner_user_prompt: str,
    png_path: pathlib.Path,
    jpg_path: pathlib.Path,
    replay_source: pathlib.Path | None = None,
) -> None:
    sticker_lines = "\n".join(
        f"{index}. {item}" for index, item in enumerate(plan["sticker_descriptions"], start=1)
    )
    replay_section = ""
    if replay_source is not None:
        replay_section = f"\n## Replay Source\n\n- Source Markdown: `{replay_source}`\n"
    planner_user_prompt_language = "json"
    planner_user_prompt_content = planner_user_prompt.strip()
    if planner_user_prompt_content:
        try:
            planner_user_prompt_content = json.dumps(
                json.loads(planner_user_prompt_content),
                indent=2,
            )
        except json.JSONDecodeError:
            planner_user_prompt_language = "text"
    else:
        planner_user_prompt_language = "text"
    style_reference_line = "-"
    if style_reference_image is not None:
        style_reference_line = str(style_reference_image.resolve())
    content = f"""# Sticker Sheet Reproduction Log

## Inputs

- Provider: {provider}
- Theme: {theme}
- Style: {style}
- Style reference image: {style_reference_line}
- Stickers per page: {stickers_per_page}
- Footer text: {FOOTER_TEXT}

## Models

- Planner model: {planner_model}
- Image model: {image_model}

## Outputs

- Transparent PNG: `{png_path.name}`
- JPG export: `{jpg_path.name}`
{replay_section}

## Sticker Descriptions

{sticker_lines}

## Planner System Prompt

```text
{planner_system_prompt}
```

## Planner User Prompt

```{planner_user_prompt_language}
{planner_user_prompt_content}
```

## Generated Image Prompt

```text
{plan["image_prompt"]}
```
"""
    log_path.write_text(content)


def ensure_tooling() -> None:
    for tool in ("swift", "sips"):
        if not shutil_which(tool):
            raise SystemExit(f"Required tool not found in PATH: {tool}")


def shutil_which(tool: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        candidate = pathlib.Path(directory) / tool
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def main() -> int:
    args = parse_args()
    ensure_tooling()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    replay_source: pathlib.Path | None = None
    planner_system_prompt = ""
    planner_user_prompt = ""
    style_reference_image: pathlib.Path | None = resolve_reference_image_path(args.style_reference_image)
    if args.replay_from_md:
        replay_source = pathlib.Path(args.replay_from_md).expanduser().resolve()
        if not replay_source.is_file():
            raise SystemExit(f"Replay Markdown file not found: {replay_source}")
        (
            plan,
            theme,
            style,
            stickers_per_page,
            replay_provider,
            replay_planner_model,
            replay_image_model,
            planner_system_prompt,
            planner_user_prompt,
            replay_style_reference_image,
        ) = load_plan_from_markdown(replay_source)
        provider = resolve_replay_provider(
            explicit_provider=args.provider,
            replay_provider=replay_provider,
            replay_planner_model=replay_planner_model,
            replay_image_model=replay_image_model,
        )
        planner_model = resolve_replay_model(
            provider=provider,
            replay_model=replay_planner_model,
            cli_model=args.planner_model,
            default_model=default_planner_model(provider),
        )
        image_model = resolve_replay_model(
            provider=provider,
            replay_model=replay_image_model,
            cli_model=args.image_model,
            default_model=default_image_model(provider),
        )
        if style_reference_image is None:
            style_reference_image = replay_style_reference_image
    else:
        if args.theme is None or args.style is None or args.stickers_per_page is None:
            raise SystemExit(
                "Provide --theme, --style, and --stickers-per-page, or use --replay-from-md."
            )
        provider = args.provider or default_provider()
        theme = args.theme
        style = args.style
        stickers_per_page = args.stickers_per_page
        planner_model = args.planner_model or default_planner_model(provider)
        image_model = args.image_model or default_image_model(provider)
        api_key = get_api_key(provider)
        plan, planner_system_prompt, planner_user_prompt = build_sticker_plan(
            provider=provider,
            api_key=api_key,
            planner_model=planner_model,
            theme=theme,
            style=style,
            stickers_per_page=stickers_per_page,
        )

    if args.replay_from_md:
        api_key = get_api_key(provider)

    if stickers_per_page < 1 or stickers_per_page > 40:
        raise SystemExit("--stickers-per-page must be between 1 and 40.")

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"{slugify(theme)}-{timestamp}"
    png_path = output_dir / f"{base_name}-a4-transparent.png"
    jpg_path = output_dir / f"{base_name}-a4.jpg"
    log_path = output_dir / f"{base_name}-prompts.md"

    raw_png_bytes = generate_image(
        provider,
        api_key,
        image_model,
        plan["image_prompt"],
        style_reference_image,
    )
    render_a4_png(raw_png_bytes, png_path, FOOTER_TEXT)
    export_jpg(png_path, jpg_path)
    write_prompt_log(
        log_path=log_path,
        provider=provider,
        theme=theme,
        style=style,
        style_reference_image=style_reference_image,
        stickers_per_page=stickers_per_page,
        planner_model=planner_model,
        image_model=image_model,
        plan=plan,
        planner_system_prompt=planner_system_prompt,
        planner_user_prompt=planner_user_prompt,
        png_path=png_path,
        jpg_path=jpg_path,
        replay_source=replay_source,
    )

    print(f"Created {png_path}")
    print(f"Created {jpg_path}")
    print(f"Created {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
