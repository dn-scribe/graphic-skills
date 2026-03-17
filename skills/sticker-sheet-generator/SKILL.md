---
name: sticker-sheet-generator
description: Create printable A4 sticker sheets from a theme prompt, a style prompt, and a sticker count. Use this when the user wants themed stickers, die-cut contours, transparent PNG output, JPG export, reproducible prompt logs, or exact footer text at the bottom of the page.
---

# Sticker Sheet Generator

Use this skill to generate a printable A4 sticker sheet with separated stickers, black cut contours, a transparent PNG, a JPG export, and a Markdown file that records the prompts and sticker plan used for reproduction.

You can also provide a sample image as a style reference during generation.

## When To Use It

Use this skill when the user asks for any of the following:

- a themed sticker sheet
- a sticker page with a transparent background
- printable A4 sticker output
- a specific number of stickers per page
- a reproducible prompt log for image generation
- exact footer text: `Nachala, the one and only!`
- OpenAI or Gemini / Nano Banana image generation

## Workflow

1. Collect three required inputs:
   - `theme`: may be a short theme or a long prompt string
   - `style`: may be a short style or a long prompt string
   - `stickers-per-page`: exact sticker count
2. If needed, install local prerequisites:

```bash
bash skills/sticker-sheet-generator/scripts/install.sh
```

3. Run the bundled script:

```bash
python3 skills/sticker-sheet-generator/scripts/generate_sticker_sheet.py \
  --theme 'Chanuka holiday in blue and gold, warm candlelight, joyful family mood' \
  --style "polished children's book illustration, premium die-cut vinyl sticker look, bold readable silhouettes" \
  --stickers-per-page 20 \
  --output-dir tmp
```

4. Confirm the outputs:
   - `<theme-slug>-<timestamp>-a4-transparent.png`
   - `<theme-slug>-<timestamp>-a4.jpg`
   - `<theme-slug>-<timestamp>-prompts.md`

To use a local style reference image:

```bash
python3 skills/sticker-sheet-generator/scripts/generate_sticker_sheet.py \
  --theme 'Chanuka holiday in blue and gold, warm candlelight, joyful family mood' \
  --style "polished children's book illustration, premium die-cut vinyl sticker look, bold readable silhouettes" \
  --style-reference-image /absolute/path/to/style-reference.png \
  --stickers-per-page 20 \
  --output-dir tmp
```

To use Gemini / Nano Banana instead of OpenAI:

```bash
python3 skills/sticker-sheet-generator/scripts/generate_sticker_sheet.py \
  --provider gemini \
  --theme 'Chanuka holiday in blue and gold, warm candlelight, joyful family mood' \
  --style "polished children's book illustration, premium die-cut vinyl sticker look, bold readable silhouettes" \
  --stickers-per-page 20 \
  --output-dir tmp
```

Gemini also supports the same `--style-reference-image` option.

## Replay From Markdown

If you want to iterate from a previous run, edit the generated Markdown file and rerun with:

```bash
python3 skills/sticker-sheet-generator/scripts/generate_sticker_sheet.py \
  --replay-from-md tmp/your-theme-20260316-120000-prompts.md \
  --output-dir tmp
```

Editable sections that affect the rerun:

- `Theme`
- `Style`
- `Style reference image`
- `Stickers per page`
- `Provider`
- `Sticker Descriptions`
- `Generated Image Prompt`
- `Planner model`
- `Image model`

If `Generated Image Prompt` is present, the script uses it directly. If you remove that block, the script rebuilds the image prompt from the edited theme, style, and sticker descriptions.

## What The Script Does

- Uses a text model to turn the theme and style inputs into:
  - an exact sticker list
  - a detailed image-generation prompt
- Uses the image model to generate a transparent portrait sticker sheet
- Optionally uses a local reference image for style guidance during generation
- Composites the result onto an A4 transparent canvas
- Adds the exact footer text `Nachala, the one and only!` at the bottom
- Exports a flattened JPG version
- Writes a Markdown log with the user inputs, generated sticker descriptions, model names, and prompts used
- Supports `--provider openai` and `--provider gemini`
- Defaults to Gemini automatically when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is present

## Requirements

- `OPENAI_API_KEY` or `OPEN_AI_TOKEN` must be set for OpenAI runs
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` must be set for Gemini runs
- This skill currently targets macOS because it relies on `swift` and `sips`
- `scripts/install.sh` bootstraps Homebrew and Python 3 if needed, then verifies the Apple toolchain

## Notes

- The footer text is added deterministically after image generation so the final file contains the exact sentence.
- The transparent background is preserved in the PNG output only. The JPG is a flattened export.
- If you omit `--provider`, the script prefers Gemini whenever a Gemini/Google API key is available in the environment.
- If the user wants revisions, rerun the script with adjusted `theme`, `style`, or `stickers-per-page`.
