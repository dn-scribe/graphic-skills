#!/usr/bin/env python3
"""Generate printable blow-pen art template pages from theme/style prompts."""

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
IMAGE_EDITS_URL = "https://api.openai.com/v1/images/edits"
FILES_URL = "https://api.openai.com/v1/files"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_PLANNER_MODEL = "gpt-4.1"
DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-1.5"
DEFAULT_GEMINI_PLANNER_MODEL = "gemini-2.5-pro"
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-3-pro-image-preview"
DEFAULT_PAGES = 4
DEFAULT_STYLE = (
    "pure black and white template art, thick black outlines only, no gray colors, "
    "no shading, ultra-minimal design, bold simple decorative border elements only, "
    "large empty white center area, suitable for blow pen art activity for children"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create blow-pen art template pages with minimal border decorations.",
    )
    parser.add_argument("--theme", help="Theme prompt or description.")
    parser.add_argument("--style", help="Style prompt or description.")
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
        "--template-image",
        help="Optional local image path to use as a template style reference during generation.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "gemini"),
        help=(
            "API provider to use for planning and image generation. "
            "If omitted, Gemini is preferred when a Gemini key is available; otherwise OpenAI is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="tmp",
        help="Directory for output files. Defaults to ./tmp",
    )
    parser.add_argument(
        "--planner-model",
        help="Text model used to plan page descriptions. Provider-specific default is used if omitted.",
    )
    parser.add_argument(
        "--image-model",
        help="Image model used to render the pages. Provider-specific default is used if omitted.",
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
        raise SystemExit(f"Template reference image not found: {path}")
    return path


def guess_mime_type(image_path: pathlib.Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type or not mime_type.startswith("image/"):
        raise SystemExit(f"Unsupported or unknown image type for template reference: {image_path}")
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
    return (slug[:48].rstrip("-")) or "blow-pen-art"


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


def planner_messages(theme: str, style: str, pages: int) -> tuple[str, str]:
    system_prompt = (
        "You are a blow-pen art template planner. Return valid JSON only. "
        "Create exactly the requested number of distinct page descriptions for blow-pen art templates. "
        "Each template page must have graphic elements ONLY at the borders and corners, "
        "leaving a large empty white area in the center for children to blow paint into. "
        "Elements must be very minimal: only a few simple decorative shapes near the page edges. "
        "All art must be pure black and white line art, absolutely no shading, fills, or gray tones. "
        "Do not include markdown fences."
    )
    user_payload = {
        "task": "Plan printable blow-pen art template pages",
        "requirements": {
            "theme": theme,
            "style": style,
            "pages": pages,
            "layout": (
                "Decorative graphic elements placed ONLY near the borders, corners, and edges. "
                "The CENTER of each page must remain completely empty and white for blow pen art. "
                "Elements should feel like a loose decorative frame or partial border."
            ),
            "element_count": (
                "Use a very small number of graphic elements per page (5-12 elements total), "
                "spread around the page borders and corners, merged with or touching the page edges."
            ),
            "art_style": (
                "Pure black and white line art with thick black outlines only. "
                "Absolutely no gray colors, no shading, no fills, no gradients."
            ),
            "purpose": (
                "Children blow paint through stencils onto the template. "
                "The empty center area receives the blown color. "
                "Keep designs very simple so young children can use them easily."
            ),
        },
        "output_schema": {
            "theme_title": "short human-readable title for the template set",
            "page_descriptions": [
                f"array of exactly {pages} distinct page concepts with specific border decoration descriptions"
            ],
            "base_prompt": (
                "one detailed style prompt used for all pages to maintain consistency, "
                "emphasizing ultra-minimal border-only decorations, large empty center, "
                "pure black and white line art"
            ),
        },
    }
    return system_prompt, json.dumps(user_payload, ensure_ascii=True)


def build_blow_pen_plan_openai(
    api_key: str,
    planner_model: str,
    theme: str,
    style: str,
    pages: int,
) -> tuple[dict, str, str]:
    system_prompt, user_prompt = planner_messages(theme, style, pages)
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

    page_descriptions = plan.get("page_descriptions")
    if not isinstance(page_descriptions, list) or not all(
        isinstance(item, str) and item.strip() for item in page_descriptions
    ):
        raise SystemExit(f"Planner returned invalid page descriptions:\n{json.dumps(plan, indent=2)}")
    if len(page_descriptions) != pages:
        raise SystemExit(
            f"Planner returned {len(page_descriptions)} page descriptions; expected {pages}."
        )

    base_prompt = plan.get("base_prompt")
    if not isinstance(base_prompt, str) or not base_prompt.strip():
        plan["base_prompt"] = (
            f"Blow-pen art template. {style}. Pure black and white line art, "
            "thick black outlines only, no gray, no shading. "
            "Decorative elements ONLY at borders and corners. Large empty white center."
        )

    theme_title = plan.get("theme_title")
    if not isinstance(theme_title, str) or not theme_title.strip():
        plan["theme_title"] = theme[:80].strip()

    return plan, system_prompt, user_prompt


def build_blow_pen_plan_gemini(
    api_key: str,
    planner_model: str,
    theme: str,
    style: str,
    pages: int,
) -> tuple[dict, str, str]:
    system_prompt, user_prompt = planner_messages(theme, style, pages)
    schema = {
        "type": "object",
        "properties": {
            "theme_title": {"type": "string"},
            "page_descriptions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": pages,
                "maxItems": pages,
            },
            "base_prompt": {"type": "string"},
        },
        "required": ["theme_title", "page_descriptions", "base_prompt"],
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

    page_descriptions = plan.get("page_descriptions")
    if not isinstance(page_descriptions, list) or not all(
        isinstance(item, str) and item.strip() for item in page_descriptions
    ):
        raise SystemExit(
            f"Gemini planner returned invalid page descriptions:\n{json.dumps(plan, indent=2)}"
        )
    if len(page_descriptions) != pages:
        raise SystemExit(
            f"Gemini planner returned {len(page_descriptions)} descriptions; expected {pages}."
        )

    base_prompt = plan.get("base_prompt")
    if not isinstance(base_prompt, str) or not base_prompt.strip():
        plan["base_prompt"] = (
            f"Blow-pen art template. {style}. Pure black and white line art, "
            "thick black outlines only, no gray, no shading. "
            "Decorative elements ONLY at borders and corners. Large empty white center."
        )

    theme_title = plan.get("theme_title")
    if not isinstance(theme_title, str) or not theme_title.strip():
        plan["theme_title"] = theme[:80].strip()

    return plan, system_prompt, user_prompt


def build_blow_pen_plan(
    provider: str,
    api_key: str,
    planner_model: str,
    theme: str,
    style: str,
    pages: int,
) -> tuple[dict, str, str]:
    if provider == "gemini":
        return build_blow_pen_plan_gemini(
            api_key=api_key,
            planner_model=planner_model,
            theme=theme,
            style=style,
            pages=pages,
        )
    return build_blow_pen_plan_openai(
        api_key=api_key,
        planner_model=planner_model,
        theme=theme,
        style=style,
        pages=pages,
    )


def build_blow_pen_prompt(prompt: str) -> str:
    return (
        "Blow-pen art template page. Pure black and white only, absolutely no colors, no gray. "
        f"{prompt}. "
        "CRITICAL LAYOUT: Place decorative graphic elements ONLY at the borders, corners, and "
        "edges of the page. The CENTER of the page must be completely empty and white. "
        "Use only 5-12 very simple decorative elements total (small outlines of shapes, leaves, "
        "stars, simple geometric forms). Elements must be near or touching the page edges, "
        "creating a loose border or frame effect. "
        "All lines must be thick black outlines only. "
        "No fills, no shading, no gray tones, no gradients. "
        "The overall design must look very sparse and minimalist. "
        "This template is used for children's blow pen art activity where paint is blown "
        "into the large empty center area."
    )


def generate_image_openai(
    api_key: str,
    image_model: str,
    image_prompt: str,
    template_ref_file_id: str | None = None,
) -> tuple[bytes | None, str | None]:
    blow_pen_prompt = build_blow_pen_prompt(image_prompt)
    try:
        if template_ref_file_id is not None:
            payload = {
                "model": image_model,
                "prompt": (
                    f"{blow_pen_prompt}\n\n"
                    "Use the attached reference image for style and decoration guidance only. "
                    "Do not copy its exact subjects."
                ),
                "images": [{"file_id": template_ref_file_id}],
                "input_fidelity": "high",
                "background": "transparent",
                "output_format": "png",
                "quality": "high",
                "size": "1024x1536",
            }
            response = post_json(IMAGE_EDITS_URL, payload, api_key)
        else:
            payload = {
                "model": image_model,
                "prompt": blow_pen_prompt,
                "background": "transparent",
                "output_format": "png",
                "quality": "high",
                "size": "1024x1536",
            }
            response = post_json(IMAGES_URL, payload, api_key)

        image_b64 = response["data"][0]["b64_json"]
        return base64.b64decode(image_b64), None
    except SystemExit as exc:
        error_msg = str(exc)
        if "content_policy_violation" in error_msg or "content filters" in error_msg:
            return None, "Content policy violation - image description may contain restricted content"
        return None, f"Generation failed: {error_msg}"
    except (KeyError, IndexError, TypeError) as exc:
        return None, f"Unexpected response format: {exc}"


def generate_image_gemini(
    api_key: str,
    image_model: str,
    image_prompt: str,
    template_ref_image_path: pathlib.Path | None = None,
) -> tuple[bytes | None, str | None]:
    blow_pen_prompt = build_blow_pen_prompt(image_prompt)
    parts: list[dict] = [{"text": blow_pen_prompt}]
    if template_ref_image_path is not None:
        parts.append(
            {
                "inlineData": {
                    "mimeType": guess_mime_type(template_ref_image_path),
                    "data": base64.b64encode(template_ref_image_path.read_bytes()).decode("utf-8"),
                }
            }
        )
        parts[0] = {
            "text": (
                f"{blow_pen_prompt}\n\n"
                "Use the attached reference image for style and decoration guidance only. "
                "Do not copy its exact subjects."
            )
        }

    payload = {"contents": [{"parts": parts}]}
    try:
        response = post_gemini_json(image_model, payload, api_key)
        candidates = response.get("candidates", [])
        for candidate in candidates:
            for part in candidate.get("content", {}).get("parts", []):
                inline_data = part.get("inlineData")
                if inline_data and inline_data.get("data"):
                    return base64.b64decode(inline_data["data"]), None
        return None, f"No image data in Gemini response: {json.dumps(response, indent=2)}"
    except SystemExit as exc:
        return None, f"Generation failed: {exc}"


def generate_image(
    provider: str,
    api_key: str,
    image_model: str,
    image_prompt: str,
    template_ref_file_id: str | None = None,
    template_ref_image_path: pathlib.Path | None = None,
) -> tuple[bytes | None, str | None]:
    if provider == "gemini":
        return generate_image_gemini(api_key, image_model, image_prompt, template_ref_image_path)
    return generate_image_openai(api_key, image_model, image_prompt, template_ref_file_id)


def generate_single_page(
    page_info: tuple,
) -> tuple[int, pathlib.Path | None, dict | None]:
    (
        i, description, base_prompt, provider, api_key, image_model,
        base_name, template_ref_file_id, template_ref_image_path, output_dir, page_num,
    ) = page_info

    page_file = output_dir / f"{base_name}-page-{page_num}.jpg"
    page_prompt = f"{base_prompt}. Template {i}: {description}"

    print(f"Generating page {i}: {description[:100]}{'...' if len(description) > 100 else ''}")
    raw_image_bytes, error_msg = generate_image(
        provider, api_key, image_model, page_prompt,
        template_ref_file_id=template_ref_file_id,
        template_ref_image_path=template_ref_image_path,
    )

    if raw_image_bytes is None:
        failure_info = {"page": i, "description": description, "error": error_msg}
        print(f"⚠️  Page {i} failed to generate: {error_msg}")
        return i, None, failure_info

    convert_to_jpg(raw_image_bytes, page_file)
    print(f"✅ Created {page_file}")
    return i, page_file, None


def convert_to_jpg(png_bytes: bytes, output_path: pathlib.Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".png") as temp_png:
        temp_png.write(png_bytes)
        temp_png.flush()
        result = subprocess.run(
            ["sips", "-s", "format", "jpeg", temp_png.name, "--out", str(output_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(f"sips conversion failed: {result.stderr}")


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


def extract_page_descriptions(markdown_text: str) -> list[str]:
    section_pattern = re.compile(
        r"^## Page Descriptions\n\n(.*?)(?=^## |\Z)",
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
    template_image = extract_bullet_value(markdown_text, "Template image")
    pages_value = extract_bullet_value(markdown_text, "Pages")
    planner_model = extract_bullet_value(markdown_text, "Planner model")
    image_model = extract_bullet_value(markdown_text, "Image model")
    base_prompt = extract_fenced_block(markdown_text, "Base Style Prompt")
    planner_system_prompt = extract_fenced_block(markdown_text, "Planner System Prompt") or ""
    planner_user_prompt = extract_fenced_block(markdown_text, "Planner User Prompt") or ""
    page_descriptions = extract_page_descriptions(markdown_text)

    if not theme or not style or not pages_value:
        raise SystemExit(
            f"Replay Markdown is missing one of Theme, Style, or Pages: {markdown_path}"
        )

    try:
        pages = int(pages_value)
    except ValueError as exc:
        raise SystemExit(f"Invalid Pages value in {markdown_path}: {pages_value}") from exc

    if not page_descriptions:
        raise SystemExit(f"Replay Markdown is missing page descriptions: {markdown_path}")

    if len(page_descriptions) != pages:
        raise SystemExit(
            f"Replay Markdown has {len(page_descriptions)} page descriptions but "
            f"Pages is {pages}."
        )

    plan = {
        "theme_title": theme[:80].strip(),
        "page_descriptions": page_descriptions,
        "base_prompt": base_prompt or (
            f"Blow-pen art template. {style}. "
            "Pure black and white line art, thick black outlines only. "
            "Decorative elements ONLY at borders and corners. Large empty white center."
        ),
    }
    resolved_provider = provider or default_provider()
    return (
        plan,
        theme,
        style,
        pages,
        resolved_provider,
        planner_model or default_planner_model(resolved_provider),
        image_model or default_image_model(resolved_provider),
        planner_system_prompt,
        planner_user_prompt,
        resolve_reference_image_path(template_image),
    )


def write_plan_log(
    log_path: pathlib.Path,
    provider: str,
    theme: str,
    style: str,
    pages: int,
    template_image: pathlib.Path | None,
    planner_model: str,
    image_model: str,
    plan: dict,
    planner_system_prompt: str,
    planner_user_prompt: str,
    page_files: list[pathlib.Path],
    failed_pages: list[dict],
    replay_source: pathlib.Path | None,
) -> None:
    template_image_text = str(template_image.resolve()) if template_image else "-"
    page_lines = "\n".join(
        f"{i}. {desc}" for i, desc in enumerate(plan["page_descriptions"], start=1)
    )

    files_section = ""
    if page_files:
        files_section += "## Generated Files\n\n"
        files_section += "\n".join(f"- {f.name}" for f in page_files)
        files_section += "\n\n"
    if failed_pages:
        files_section += "## Failed Pages\n\n"
        for failure in failed_pages:
            files_section += f"- **Page {failure['page']}**: {failure['error']}\n"
            files_section += f"  - Description: {failure['description']}\n"
        files_section += "\n"

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

    content = f"""# Blow Pen Art Reproduction Log

## Inputs

- Provider: {provider}
- Theme: {theme}
- Style: {style}
- Template image: {template_image_text}
- Pages: {pages}

## Models

- Planner model: {planner_model}
- Image model: {image_model}

{files_section}{replay_section}
## Page Descriptions

{page_lines}

## Base Style Prompt

```text
{plan["base_prompt"]}
```

## Planner System Prompt

```text
{planner_system_prompt}
```

## Planner User Prompt

```{planner_user_prompt_language}
{planner_user_prompt_content}
```
"""
    log_path.write_text(content)


def shutil_which(tool: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        candidate = pathlib.Path(directory) / tool
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def ensure_tooling() -> None:
    if not shutil_which("sips"):
        raise SystemExit("Required tool not found in PATH: sips")


def main() -> int:
    args = parse_args()
    ensure_tooling()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    replay_source: pathlib.Path | None = None
    planner_system_prompt = ""
    planner_user_prompt = ""
    template_image: pathlib.Path | None = resolve_reference_image_path(args.template_image)

    if args.replay_from_md:
        replay_source = pathlib.Path(args.replay_from_md).expanduser().resolve()
        if not replay_source.is_file():
            raise SystemExit(f"Replay Markdown file not found: {replay_source}")
        (
            plan,
            theme,
            style,
            pages,
            replay_provider,
            replay_planner_model,
            replay_image_model,
            planner_system_prompt,
            planner_user_prompt,
            replay_template_image,
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
        if template_image is None:
            template_image = replay_template_image
    else:
        if args.theme is None:
            raise SystemExit("Provide --theme or use --replay-from-md.")
        provider = args.provider or default_provider()
        theme = args.theme
        style = args.style or DEFAULT_STYLE
        pages = args.pages
        planner_model = args.planner_model or default_planner_model(provider)
        image_model = args.image_model or default_image_model(provider)
        api_key = get_api_key(provider)
        plan, planner_system_prompt, planner_user_prompt = build_blow_pen_plan(
            provider=provider,
            api_key=api_key,
            planner_model=planner_model,
            theme=theme,
            style=style,
            pages=pages,
        )

    if args.replay_from_md:
        api_key = get_api_key(provider)

    if pages < 1 or pages > 20:
        raise SystemExit("--pages must be between 1 and 20.")

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"{slugify(theme)}-{timestamp}"
    log_path = output_dir / f"{base_name}-plan.md"

    # Upload template reference image once (OpenAI) to reuse across all pages.
    template_ref_file_id: str | None = None
    if template_image is not None and provider == "openai":
        print(f"Uploading template reference image: {template_image.name}")
        template_ref_file_id = upload_openai_file(api_key, template_image)

    page_files: list[pathlib.Path] = []
    failed_pages: list[dict] = []

    page_info_list = []
    for i, description in enumerate(plan["page_descriptions"], 1):
        page_num = f"{i:03d}"
        page_info = (
            i, description, plan["base_prompt"], provider, api_key, image_model,
            base_name, template_ref_file_id,
            template_image if provider == "gemini" else None,
            output_dir, page_num,
        )
        page_info_list.append(page_info)

    # Generate first page, then remaining in parallel.
    _, page_file_1, failure_1 = generate_single_page(page_info_list[0])
    if page_file_1 is not None:
        page_files.append(page_file_1)
    elif failure_1 is not None:
        failed_pages.append(failure_1)

    if len(page_info_list) > 1:
        remaining = page_info_list[1:]
        print(f"Generating remaining {len(remaining)} pages in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(generate_single_page, info) for info in remaining]
            for future in concurrent.futures.as_completed(futures):
                try:
                    _, page_file, failure_info = future.result()
                    if page_file is not None:
                        page_files.append(page_file)
                    elif failure_info is not None:
                        failed_pages.append(failure_info)
                except Exception as exc:
                    print(f"⚠️  Page generation error: {exc}")
                    failed_pages.append({
                        "page": "unknown",
                        "description": "unknown",
                        "error": f"Parallel generation error: {exc}",
                    })

    page_files.sort(key=lambda p: p.name)

    write_plan_log(
        log_path=log_path,
        provider=provider,
        theme=theme,
        style=style,
        pages=pages,
        template_image=template_image,
        planner_model=planner_model,
        image_model=image_model,
        plan=plan,
        planner_system_prompt=planner_system_prompt,
        planner_user_prompt=planner_user_prompt,
        page_files=page_files,
        failed_pages=failed_pages,
        replay_source=replay_source,
    )

    print(f"📝 Created {log_path}")

    successful_count = len(page_files)
    failed_count = len(failed_pages)

    if successful_count > 0:
        print(f"✅ Generated {successful_count} blow-pen art template pages successfully")

    if failed_count > 0:
        print(f"❌ {failed_count} pages failed to generate:")
        for failure in failed_pages:
            print(f"   • Page {failure['page']}: {failure['error']}")
        print("\n💡 Tip: Try rephrasing problematic descriptions to avoid content filters")

    return 0 if successful_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
