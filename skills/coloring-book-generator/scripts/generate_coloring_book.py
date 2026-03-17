#!/usr/bin/env python3
"""Generate printable coloring book pages from theme/style prompts."""

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
DEFAULT_OPENAI_PLANNER_MODEL = "gpt-4o"
DEFAULT_OPENAI_IMAGE_MODEL = "dall-e-3"
DEFAULT_GEMINI_PLANNER_MODEL = "gemini-1.5-pro"
DEFAULT_GEMINI_IMAGE_MODEL = "dall-e-3"  # Fallback to OpenAI for image generation
DEFAULT_PAGES = 5
DEFAULT_STYLE = "pure black and white coloring book line art with thick black outlines only, no gray colors, no shading, minimal detail, suitable for children"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create coloring book pages with black and white line art.",
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
        "--reference-image",
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
        help="Text model used to plan picture descriptions. Provider-specific default is used if omitted.",
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
    # Default to OpenAI since it supports both planning and image generation reliably
    return "openai"


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
        raise SystemExit(f"Reference image not found: {path}")
    return path


def guess_mime_type(image_path: pathlib.Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type or not mime_type.startswith("image/"):
        raise SystemExit(f"Unsupported or unknown image type for reference: {image_path}")
    return mime_type


def slugify(value: str) -> str:
    slug = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return (slug[:48].rstrip("-")) or "coloring-book"


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
        # Re-raise with more information for better error handling downstream
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


def planner_messages(theme: str, style: str, pages: int) -> tuple[str, str]:
    system_prompt = (
        "You are a coloring book planner. Return valid JSON only. "
        "Create exactly the requested number of distinct picture descriptions for coloring book pages. "
        "Each page should be suitable for PURE BLACK AND WHITE line art with thick black outlines only. "
        "NO gray colors, NO shading, NO gradients - only pure black lines on white background. "
        "Do not include markdown fences."
    )
    user_payload = {
        "task": "Plan a printable coloring book",
        "requirements": {
            "theme": theme,
            "style": style,
            "pages": pages,
            "art_style": "pure black and white line art with thick black outlines only, absolutely no gray colors or shading",
            "complexity": "appropriate detail level for coloring activities",
            "consistency": "maintain characters and style across pages for story continuity",
        },
        "output_schema": {
            "theme_title": "short human-readable title",
            "picture_descriptions": [
                f"array of exactly {pages} distinct picture concepts that work well as coloring pages"
            ],
            "base_prompt": (
                "one detailed style prompt that will be used for all images to maintain consistency, "
                "focusing on PURE black and white line art with thick black outlines only, no gray colors, no shading"
            ),
        },
    }
    return system_prompt, json.dumps(user_payload, ensure_ascii=True)


def build_coloring_plan_openai(
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

    picture_descriptions = plan.get("picture_descriptions")
    if not isinstance(picture_descriptions, list) or not all(
        isinstance(item, str) and item.strip() for item in picture_descriptions
    ):
        raise SystemExit(f"Planner returned invalid picture descriptions:\n{json.dumps(plan, indent=2)}")
    if len(picture_descriptions) != pages:
        raise SystemExit(
            f"Planner returned {len(picture_descriptions)} pictures; expected {pages}."
        )

    base_prompt = plan.get("base_prompt")
    if not isinstance(base_prompt, str) or not base_prompt.strip():
        plan["base_prompt"] = f"Pure black and white coloring book style line art. {style}. Thick black outlines only, no gray colors, no shading, minimal detail, suitable for coloring."

    theme_title = plan.get("theme_title")
    if not isinstance(theme_title, str) or not theme_title.strip():
        plan["theme_title"] = theme[:80].strip()

    return plan, system_prompt, user_prompt


def build_coloring_plan_gemini(
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
            "picture_descriptions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": pages,
                "maxItems": pages,
            },
            "base_prompt": {"type": "string"},
        },
        "required": ["theme_title", "picture_descriptions", "base_prompt"],
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


def build_coloring_plan(
    provider: str,
    api_key: str,
    planner_model: str,
    theme: str,
    style: str,
    pages: int,
) -> tuple[dict, str, str]:
    if provider == "openai":
        return build_coloring_plan_openai(api_key, planner_model, theme, style, pages)
    return build_coloring_plan_gemini(api_key, planner_model, theme, style, pages)


def generate_image_openai(api_key: str, image_model: str, prompt: str, reference_image: pathlib.Path | None) -> tuple[bytes | None, str | None]:
    # Add coloring book specific prompt additions with emphasis on pure black and white
    coloring_prompt = f"Pure black and white line art coloring book page. {prompt}. IMPORTANT: Only pure black lines on white background, absolutely no gray colors, no shading, no gradients, thick black outlines only, simple design suitable for children to color in."
    
    if reference_image:
        coloring_prompt = f"{coloring_prompt} (Use similar style and composition as reference image provided)"
    
    payload = {
        "model": image_model,
        "prompt": coloring_prompt,
        "n": 1,
        "size": "1024x1024",
        "response_format": "b64_json",
    }
    
    try:
        response = post_json(IMAGES_URL, payload, api_key)
        b64_data = response["data"][0]["b64_json"]
        return base64.b64decode(b64_data), None
    except SystemExit as exc:
        # Check if it's a content policy violation or other API error
        error_msg = str(exc)
        if "content_policy_violation" in error_msg or "content filters" in error_msg:
            return None, "Content policy violation - image description may contain restricted content"
        elif "400" in error_msg:
            return None, f"API error: {error_msg}"
        else:
            return None, f"Generation failed: {error_msg}"
    except (KeyError, IndexError, TypeError) as exc:
        return None, f"Unexpected response format: {exc}"


def generate_image_gemini(api_key: str, image_model: str, prompt: str, reference_image: pathlib.Path | None) -> tuple[bytes | None, str | None]:
    # Gemini doesn't support image generation, so we fall back to OpenAI
    # But we need an OpenAI API key for this
    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_AI_TOKEN")
    if not openai_key:
        return None, "Gemini doesn't support image generation and no OpenAI API key available for fallback"
    
    print("Note: Using OpenAI for image generation since Gemini doesn't support it.")
    return generate_image_openai(openai_key, "dall-e-3", prompt, reference_image)


def generate_image(provider: str, api_key: str, image_model: str, prompt: str, reference_image: pathlib.Path | None = None) -> tuple[bytes | None, str | None]:
    if provider == "openai":
        return generate_image_openai(api_key, image_model, prompt, reference_image)
    return generate_image_gemini(api_key, image_model, prompt, reference_image)


def generate_single_page(
    page_info: tuple[int, str, str, str, str, str, str, pathlib.Path | None, pathlib.Path, str]
) -> tuple[int, pathlib.Path | None, dict | None]:
    """Generate a single page. Returns (page_number, page_file_path, failure_info)"""
    (i, description, base_prompt, provider, api_key, image_model, base_name, 
     reference_image, output_dir, page_num) = page_info
    
    page_file = output_dir / f"{base_name}-page-{page_num}.jpg"
    
    # Create detailed prompt for this page
    page_prompt = f"{base_prompt}. Page {i}: {description}"
    
    print(f"Generating page {i}: {description[:100]}{'...' if len(description) > 100 else ''}")
    raw_image_bytes, error_msg = generate_image(provider, api_key, image_model, page_prompt, reference_image)
    
    if raw_image_bytes is None:
        # Page generation failed
        failure_info = {
            "page": i,
            "description": description,
            "error": error_msg
        }
        print(f"⚠️  Page {i} failed to generate: {error_msg}")
        return i, None, failure_info
    
    # Convert to JPG and save
    convert_to_jpg(raw_image_bytes, page_file)
    print(f"✅ Created {page_file}")
    return i, page_file, None


def convert_to_jpg(png_bytes: bytes, output_path: pathlib.Path) -> None:
    """Convert PNG bytes to JPG file using sips."""
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


def load_plan_from_markdown(md_path: pathlib.Path) -> tuple[
    dict, str, str, int, str | None, str | None, str | None, str, str, pathlib.Path | None
]:
    """Load plan from markdown file for replay functionality."""
    content = md_path.read_text()
    
    # Extract basic fields
    theme = ""
    style = ""
    pages = 5
    provider = None
    planner_model = None
    image_model = None
    reference_image = None
    planner_system_prompt = ""
    planner_user_prompt = ""
    
    # Simple regex-based extraction (could be more robust)
    if theme_match := re.search(r"Theme:\s*(.+)", content):
        theme = theme_match.group(1).strip()
    if style_match := re.search(r"Style:\s*(.+)", content):
        style = style_match.group(1).strip()
    if pages_match := re.search(r"Number of pages:\s*(\d+)", content):
        pages = int(pages_match.group(1))
    if provider_match := re.search(r"Provider:\s*(.+)", content):
        provider = provider_match.group(1).strip()
    if planner_match := re.search(r"Planner model:\s*(.+)", content):
        planner_model = planner_match.group(1).strip()
    if image_match := re.search(r"Image model:\s*(.+)", content):
        image_model = image_match.group(1).strip()
    if ref_match := re.search(r"Reference image:\s*(.+)", content):
        ref_path = ref_match.group(1).strip()
        if ref_path != "None" and ref_path != "-":
            reference_image = pathlib.Path(ref_path)
    
    # Extract picture descriptions from markdown
    picture_descriptions = []
    desc_section = re.search(r"## Picture Descriptions\s*(.+?)(?=##|$)", content, re.DOTALL)
    if desc_section:
        for line in desc_section.group(1).strip().split('\n'):
            line = line.strip()
            if line.startswith('- '):
                picture_descriptions.append(line[2:])
    
    # Build basic plan structure
    plan = {
        "theme_title": theme[:80].strip(),
        "picture_descriptions": picture_descriptions,
        "base_prompt": f"Black and white coloring book style line art. {style}. Thick black outlines, minimal detail, suitable for coloring."
    }
    
    return plan, theme, style, pages, provider, planner_model, image_model, planner_system_prompt, planner_user_prompt, reference_image


def write_plan_log(
    log_path: pathlib.Path,
    provider: str,
    theme: str,
    style: str,
    pages: int,
    reference_image: pathlib.Path | None,
    planner_model: str,
    image_model: str,
    plan: dict,
    planner_system_prompt: str,
    planner_user_prompt: str,
    page_files: list[pathlib.Path],
    failed_pages: list[dict],
    replay_source: pathlib.Path | None,
) -> None:
    """Write a markdown log of the generation process."""
    reference_text = str(reference_image) if reference_image else "None"
    
    # Generate files section
    files_section = ""
    if page_files:
        files_section += "## Successfully Generated Files\n\n"
        files_section += "\n".join(f"- {page_file.name}" for page_file in page_files)
        files_section += "\n\n"
    
    if failed_pages:
        files_section += "## Failed Pages\n\n"
        for failure in failed_pages:
            files_section += f"- **Page {failure['page']}**: {failure['error']}\n"
            files_section += f"  - Description: {failure['description']}\n"
        files_section += "\n"
    
    content = f"""# Coloring Book Generation Plan

Generated: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{"Replayed from: " + str(replay_source) if replay_source else "Original generation"}

## Inputs

- **Theme**: {theme}
- **Style**: {style}
- **Number of pages**: {pages}
- **Reference image**: {reference_text}
- **Provider**: {provider}
- **Planner model**: {planner_model}
- **Image model**: {image_model}

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


def ensure_tooling() -> None:
    """Ensure required command-line tools are available."""
    import shutil
    for tool in ("sips",):
        if not shutil.which(tool):
            raise SystemExit(f"Required tool not found in PATH: {tool}")


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
            style,
            pages,
            replay_provider,
            replay_planner_model,
            replay_image_model,
            planner_system_prompt,
            planner_user_prompt,
            replay_reference_image,
        ) = load_plan_from_markdown(replay_source)
        provider = replay_provider or args.provider or default_provider()
        planner_model = args.planner_model or replay_planner_model or default_planner_model(provider)
        image_model = args.image_model or replay_image_model or default_image_model(provider)
        if reference_image is None:
            reference_image = replay_reference_image
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
        plan, planner_system_prompt, planner_user_prompt = build_coloring_plan(
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
    
    page_files = []
    first_page_reference = None
    failed_pages = []
    
    # Generate first page to establish style reference
    first_description = plan["picture_descriptions"][0]
    page_prompt = f"{plan['base_prompt']}. Page 1: {first_description}"
    
    print(f"Generating page 1/{pages}: {first_description}")
    raw_image_bytes, error_msg = generate_image(provider, api_key, image_model, page_prompt, reference_image)
    
    if raw_image_bytes is not None:
        # First page successful - save it and use as reference
        first_page_file = output_dir / f"{base_name}-page-001.jpg"
        convert_to_jpg(raw_image_bytes, first_page_file)
        page_files.append(first_page_file)
        
        # Save as PNG reference for subsequent pages
        first_page_png = output_dir / f"{base_name}-page-001-ref.png"
        first_page_png.write_bytes(raw_image_bytes)
        first_page_reference = first_page_png
        print(f"✅ Created {first_page_file}")
    else:
        # First page failed
        failed_pages.append({
            "page": 1,
            "description": first_description,
            "error": error_msg
        })
        print(f"⚠️  Page 1 failed to generate: {error_msg}")
    
    # Generate remaining pages in parallel if there are any
    if len(plan["picture_descriptions"]) > 1:
        remaining_descriptions = plan["picture_descriptions"][1:]
        page_info_list = []
        
        for i, description in enumerate(remaining_descriptions, 2):
            page_num = f"{i:03d}"
            page_info = (
                i, description, plan['base_prompt'], provider, api_key, image_model,
                base_name, first_page_reference, output_dir, page_num
            )
            page_info_list.append(page_info)
        
        # Generate pages in parallel
        print(f"Generating remaining {len(page_info_list)} pages in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(generate_single_page, page_info) for page_info in page_info_list]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    page_num, page_file, failure_info = future.result()
                    if page_file is not None:
                        page_files.append(page_file)
                    else:
                        failed_pages.append(failure_info)
                except Exception as exc:
                    print(f"⚠️  Page generation error: {exc}")
                    # Add to failed pages if we can't determine which one failed
                    failed_pages.append({
                        "page": "unknown",
                        "description": "unknown",
                        "error": f"Parallel generation error: {exc}"
                    })
    
    # Sort page files by page number for consistent ordering
    page_files.sort(key=lambda p: p.name)

    # Write the plan log
    write_plan_log(
        log_path=log_path,
        provider=provider,
        theme=theme,
        style=style,
        pages=pages,
        reference_image=reference_image,
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
    
    # Summary
    successful_count = len(page_files)
    failed_count = len(failed_pages)
    
    if successful_count > 0:
        print(f"✅ Generated {successful_count} coloring book pages successfully")
    
    if failed_count > 0:
        print(f"❌ {failed_count} pages failed to generate:")
        for failure in failed_pages:
            print(f"   • Page {failure['page']}: {failure['error']}")
        print("\n💡 Tip: Try rephrasing problematic descriptions to avoid content filters")
    
    return 0 if successful_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())