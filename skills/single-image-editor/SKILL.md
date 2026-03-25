---
name: single-image-editor
description: Edit one existing image from a text prompt, with an optional strict reference-preserving workflow. Use when the user wants to modify a single local picture, keep most of the source composition, add or remove elements, or iterate on one edited image with OpenAI or Gemini.
---

# Single Image Editor

Use this skill to edit one local image from a prompt and save one output image plus a Markdown log of the exact request.

## When To Use It

Use this skill when the user asks for any of the following:

- edit one existing local image
- keep the structure or composition of a source image while changing part of it
- add, remove, or replace elements in a single picture
- do a strict reference-based image edit with OpenAI
- do a single image edit with Gemini
- save a reproducible prompt log for one edited image

## Workflow

1. Collect the required inputs:
   - `input_image`: absolute or repo-relative path to the source image
   - `prompt`: exact edit instruction
   - `provider`: optional, defaults to Gemini when a Gemini key is present, otherwise OpenAI
   - `image_model`: optional override
2. If needed, install local prerequisites:

```bash
bash skills/single-image-editor/scripts/install.sh
```

3. Run the bundled script:

```bash
python3 skills/single-image-editor/scripts/generate_single_image_edit.py \
  --input-image /absolute/path/to/source.jpg \
  --prompt "Keep the room layout the same, but replace the blue sofa with a brown leather sofa." \
  --output-dir tmp
```

To force OpenAI:

```bash
python3 skills/single-image-editor/scripts/generate_single_image_edit.py \
  --provider openai \
  --input-image /absolute/path/to/source.jpg \
  --prompt "Preserve the flat roof exactly. Replace the table with two children." \
  --output-dir tmp
```

To force Gemini:

```bash
python3 skills/single-image-editor/scripts/generate_single_image_edit.py \
  --provider gemini \
  --input-image /absolute/path/to/source.jpg \
  --prompt "Keep the same composition, but add a red kite in the sky." \
  --output-dir tmp
```

4. Confirm the outputs:
   - `<image-slug>-<timestamp>.<png|jpg|webp>`
   - `<image-slug>-<timestamp>-edit.md`

## Replay From Markdown

If you want to iterate from a previous run, edit the generated Markdown file and rerun with:

```bash
python3 skills/single-image-editor/scripts/generate_single_image_edit.py \
  --replay-from-md tmp/your-edit-20260325-120000-edit.md \
  --output-dir tmp
```

Editable sections that affect the rerun:

- `Input image`
- `Prompt`
- `Provider`
- `Image model`

## Notes

- OpenAI uses the image edit endpoint with the source image attached, which is the best option for strict structure-preserving edits.
- Gemini uses image-plus-text editing through `generateContent` and returns the edited image directly.
- The script stores a Markdown log with the exact prompt, provider, model, and output path.
- Keep prompts concrete when geometry matters: explicitly state what must stay unchanged and what must not happen.
