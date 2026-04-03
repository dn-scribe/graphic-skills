---
name: blow-pen-art-generator
description: Create printable blow-pen art template pages from a theme prompt, optional template reference image, style, and page count. Pages have ultra-minimal black and white border decorations with a large empty center area for children to blow paint into. Use this when the user wants blow pen art templates, sparse border-only designs, pure black and white page templates, printable blow pen activity pages, or reproducible prompt logs.
---

# Blow Pen Art Generator

Use this skill to generate printable blow-pen art template pages. Each page has simple black and white decorative elements placed only at the borders and corners, leaving a large empty center area where children blow paint through the template to create colorful art.

You can provide a picture of an existing template as a style reference, and optionally a style description to guide the decoration aesthetic.

## When To Use It

Use this skill when the user asks for any of the following:

- blow pen art templates or pages
- templates for blowing paint activities
- printable pages with border decorations and empty center
- a specific number of blow pen template pages
- black and white template pages with minimal decorations
- a reproducible prompt log for blow pen art generation
- OpenAI or Gemini / Nano Banana image generation

## Workflow

1. Collect the inputs:
   - `theme`: required — theme or description of the template decorations (e.g. "butterflies and flowers", "space rockets")
   - `style`: optional — style description (defaults to minimal black and white border art)
   - `pages`: number of pages to generate (default 4, range 1–20)
   - `template-image`: optional — a local image of an existing blow pen template for style reference
2. If needed, install local prerequisites:

```bash
bash skills/blow-pen-art-generator/scripts/install.sh
```

3. Run the bundled script:

```bash
python3 skills/blow-pen-art-generator/scripts/generate_blow_pen_art.py \
  --theme 'spring flowers and butterflies' \
  --pages 4 \
  --output-dir tmp
```

4. Confirm the outputs:
   - `<theme-slug>-<timestamp>-page-001.jpg` through `<theme-slug>-<timestamp>-page-XXX.jpg`
   - `<theme-slug>-<timestamp>-plan.md`

To use an existing template image as a style reference:

```bash
python3 skills/blow-pen-art-generator/scripts/generate_blow_pen_art.py \
  --theme 'spring flowers and butterflies' \
  --template-image /absolute/path/to/existing-template.png \
  --pages 4 \
  --output-dir tmp
```

To specify a custom style:

```bash
python3 skills/blow-pen-art-generator/scripts/generate_blow_pen_art.py \
  --theme 'ocean animals' \
  --style 'bold graphic style, thick outlines, very simple shapes' \
  --pages 6 \
  --output-dir tmp
```

To use Gemini / Nano Banana instead of OpenAI:

```bash
python3 skills/blow-pen-art-generator/scripts/generate_blow_pen_art.py \
  --provider gemini \
  --theme 'spring flowers and butterflies' \
  --pages 4 \
  --output-dir tmp
```

## Replay From Markdown

If you want to iterate from a previous run, edit the generated Markdown file and rerun with:

```bash
python3 skills/blow-pen-art-generator/scripts/generate_blow_pen_art.py \
  --replay-from-md tmp/your-theme-20260322-120000-plan.md \
  --output-dir tmp
```

Editable sections that affect the rerun:

- `Theme`
- `Style`
- `Template image`
- `Pages`
- `Provider`
- `Page Descriptions`
- `Base Style Prompt`
- `Planner model`
- `Image model`

If `Base Style Prompt` is present, the script uses it directly when building image prompts. Edit the page descriptions or base prompt to adjust the output on replay.

## What The Script Does

- Uses a text model to turn the theme and style inputs into:
  - an exact list of page descriptions, each specifying unique border decoration concepts
  - a base style prompt used for all pages to maintain visual consistency
- Uses the image model to generate each page as a portrait image with:
  - decorative elements placed only at the borders and corners
  - a large empty white center area for blow pen activity
  - pure black outlines on a white background — no colors, no gray, no shading
- Optionally uses a local template image as style guidance during generation
- Generates the first page, then remaining pages in parallel (up to 3 concurrent requests)
- Converts each generated image to JPEG using `sips`
- Writes a Markdown log with all inputs, page descriptions, model names, and prompts for reproduction
- Supports `--provider openai` and `--provider gemini`
- Defaults to Gemini automatically when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is present

## Requirements

- `OPENAI_API_KEY` or `OPEN_AI_TOKEN` must be set for OpenAI runs
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` must be set for Gemini runs
- This skill currently targets macOS because it relies on `sips`
- `scripts/install.sh` bootstraps Homebrew and Python 3 if needed, then verifies `sips`

## Notes

- Pages are intentionally very sparse — each has only 5–12 small decorative elements near the edges
- The large empty center is essential for the blow pen activity: children blow paint through stencils into this area
- All images use only pure black outlines on a white background with no fills, shading, or gray tones
- The `--template-image` option accepts any local image (PNG, JPG, etc.) to guide the visual style
- For parallel page generation, up to 3 pages are generated simultaneously after the first page
- If a page fails (e.g., content policy), the script skips it, continues, and reports failures at the end
- If the user wants revisions, edit the plan Markdown and rerun with `--replay-from-md`
