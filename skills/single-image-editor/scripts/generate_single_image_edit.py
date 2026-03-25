#!/usr/bin/env python3
"""Edit a single local image with OpenAI or Gemini and save a prompt log."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import mimetypes
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request


IMAGE_EDITS_URL = "https://api.openai.com/v1/images/edits"
FILES_URL = "https://api.openai.com/v1/files"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-1.5"
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edit one local image with OpenAI or Gemini.",
    )
    parser.add_argument("--input-image", help="Path to the source image to edit.")
    parser.add_argument("--prompt", help="Edit instruction.")
    parser.add_argument(
        "--replay-from-md",
        help="Reuse an existing edit Markdown file as editable input.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "gemini"),
        help="If omitted, Gemini is preferred when a Gemini key is available; otherwise OpenAI is used.",
    )
    parser.add_argument(
        "--image-model",
        help="Model override for the selected provider.",
    )
    parser.add_argument(
        "--output-dir",
        default="tmp",
        help="Directory for output files. Defaults to ./tmp",
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


def default_image_model(provider: str) -> str:
    if provider == "gemini":
        return DEFAULT_GEMINI_IMAGE_MODEL
    return DEFAULT_OPENAI_IMAGE_MODEL


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
    replay_image_model: str | None,
) -> str:
    if explicit_provider:
        return explicit_provider
    inferred = infer_provider_from_model(replay_image_model)
    if inferred:
        return inferred
    if replay_provider in {"openai", "gemini"}:
        return replay_provider
    return default_provider()


def resolve_replay_model(provider: str, replay_model: str | None, cli_model: str | None) -> str:
    if cli_model:
        return cli_model
    if replay_model and infer_provider_from_model(replay_model) in {None, provider}:
        return replay_model
    return default_image_model(provider)


def resolve_image_path(image_path: str | None) -> pathlib.Path:
    if not image_path:
        raise SystemExit("Provide --input-image or use --replay-from-md.")
    path = pathlib.Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Input image not found: {path}")
    return path


def guess_mime_type(image_path: pathlib.Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type or not mime_type.startswith("image/"):
        raise SystemExit(f"Unsupported or unknown image type: {image_path}")
    return mime_type


def slugify(value: str) -> str:
    slug = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return (slug[:48].rstrip("-")) or "image-edit"


def post_json(url: str, payload: dict, api_key: str, extra_headers: dict[str, str] | None = None) -> dict:
    headers = {
        "Content-Type": "application/json",
    }
    headers.update(extra_headers or {})
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
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


def post_openai_json(url: str, payload: dict, api_key: str) -> dict:
    return post_json(url, payload, api_key, {"Authorization": f"Bearer {api_key}"})


def post_gemini_json(model: str, payload: dict, api_key: str) -> dict:
    url = f"{GEMINI_API_BASE_URL}/{model}:generateContent"
    return post_json(url, payload, api_key, {"x-goog-api-key": api_key})


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


def extension_for_mime_type(mime_type: str) -> str:
    if mime_type == "image/png":
        return ".png"
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/webp":
        return ".webp"
    return ".bin"


def generate_image_openai(
    api_key: str,
    image_model: str,
    prompt: str,
    input_image: pathlib.Path,
) -> tuple[bytes, str]:
    file_id = upload_openai_file(api_key, input_image)
    payload = {
        "model": image_model,
        "prompt": prompt,
        "images": [{"file_id": file_id}],
        "input_fidelity": "high",
        "output_format": "png",
        "quality": "high",
    }
    response = post_openai_json(IMAGE_EDITS_URL, payload, api_key)
    try:
        image_b64 = response["data"][0]["b64_json"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SystemExit(f"Unexpected OpenAI image edit response: {json.dumps(response, indent=2)}") from exc
    return base64.b64decode(image_b64), "image/png"


def generate_image_gemini(
    api_key: str,
    image_model: str,
    prompt: str,
    input_image: pathlib.Path,
) -> tuple[bytes, str]:
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": guess_mime_type(input_image),
                            "data": base64.b64encode(input_image.read_bytes()).decode("utf-8"),
                        }
                    },
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
        },
    }
    response = post_gemini_json(image_model, payload, api_key)
    candidates = response.get("candidates", [])
    for candidate in candidates:
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            inline_data = part.get("inlineData")
            if inline_data and inline_data.get("data"):
                return base64.b64decode(inline_data["data"]), inline_data.get("mimeType", "image/png")
    raise SystemExit(f"Unexpected Gemini image response: {json.dumps(response, indent=2)}")


def generate_image(
    provider: str,
    api_key: str,
    image_model: str,
    prompt: str,
    input_image: pathlib.Path,
) -> tuple[bytes, str]:
    if provider == "gemini":
        return generate_image_gemini(api_key, image_model, prompt, input_image)
    return generate_image_openai(api_key, image_model, prompt, input_image)


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


def load_edit_from_markdown(
    markdown_path: pathlib.Path,
) -> tuple[pathlib.Path, str, str, str | None]:
    markdown_text = markdown_path.read_text()
    input_image_value = extract_bullet_value(markdown_text, "Input image")
    prompt = extract_fenced_block(markdown_text, "Prompt")
    provider = extract_bullet_value(markdown_text, "Provider")
    image_model = extract_bullet_value(markdown_text, "Image model")

    if not input_image_value or not prompt:
        raise SystemExit(f"Replay Markdown is missing Input image or Prompt: {markdown_path}")

    return resolve_image_path(input_image_value), prompt, provider or default_provider(), image_model


def write_edit_log(
    log_path: pathlib.Path,
    input_image: pathlib.Path,
    prompt: str,
    provider: str,
    image_model: str,
    output_image: pathlib.Path,
    replay_source: pathlib.Path | None,
) -> None:
    content = f"""# Single Image Edit Log

Generated: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{"Replayed from: " + str(replay_source) if replay_source else "Original generation"}

## Inputs

- Input image: {input_image}
- Provider: {provider}
- Image model: {image_model}
- Output image: {output_image}

## Prompt

```text
{prompt}
```
"""
    log_path.write_text(content)


def main() -> int:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    replay_source: pathlib.Path | None = None
    if args.replay_from_md:
        replay_source = pathlib.Path(args.replay_from_md).expanduser().resolve()
        if not replay_source.is_file():
            raise SystemExit(f"Replay Markdown file not found: {replay_source}")
        replay_input_image, replay_prompt, replay_provider, replay_image_model = load_edit_from_markdown(
            replay_source
        )
        input_image = resolve_image_path(args.input_image) if args.input_image else replay_input_image
        prompt = args.prompt or replay_prompt
        provider = resolve_replay_provider(args.provider, replay_provider, replay_image_model)
        image_model = resolve_replay_model(provider, replay_image_model, args.image_model)
    else:
        input_image = resolve_image_path(args.input_image)
        prompt = args.prompt
        if not prompt or not prompt.strip():
            raise SystemExit("Provide --prompt or use --replay-from-md.")
        provider = args.provider or default_provider()
        image_model = args.image_model or default_image_model(provider)

    api_key = get_api_key(provider)
    prompt = prompt.strip()
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"{slugify(input_image.stem)}-{timestamp}"
    print(f"Editing image with {provider}:{image_model}")
    raw_image_bytes, output_mime_type = generate_image(provider, api_key, image_model, prompt, input_image)
    output_image = output_dir / f"{base_name}{extension_for_mime_type(output_mime_type)}"
    log_path = output_dir / f"{base_name}-edit.md"
    output_image.write_bytes(raw_image_bytes)
    print(f"✅ Created {output_image}")

    write_edit_log(
        log_path=log_path,
        input_image=input_image,
        prompt=prompt,
        provider=provider,
        image_model=image_model,
        output_image=output_image,
        replay_source=replay_source,
    )
    print(f"📝 Created {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
