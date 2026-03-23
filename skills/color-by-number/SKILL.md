---
name: color-by-number
description: Create color-by-number activity pages from a theme, optional color set, and optional reference picture. For each page generates a full-color version and a matching black-and-white version with numbered regions ready for coloring. Use when the user wants color-by-number worksheets, activity books, or themed numbered coloring pages.
---

# Color-by-Number Generator

Use this skill to generate educational color-by-number activity books designed by child development specialists for ages **3-8 years old**. For every page the script produces two files: a full-color reference image and a matching black-and-white line-art image whose regions are numbered so the colorist knows which crayon to use.

The generator creates **5-10 large, simple regions** perfect for small hands to color, with numbers placed **inside each color area** using advanced region analysis. The colored images strictly use only the specified palette colors for clean B&W conversion.

**Educational Features:**
- **Age-appropriate design:** Simple, recognizable shapes for ages 3-8
- **Optimal complexity:** 5-10 distinct sections maximum per page
- **Motor skill development:** Large regions sized for small hands
- **Learning benefits:** Number recognition, color matching, fine motor skills
- **Quality controlled:** High contrast, clear boundaries, child-friendly themes

You can supply your own color palette or let the script fall back to a standard 6-color crayon set optimized for young children.

## When To Use It

Use this skill when the user asks for any of the following:

- a color-by-number activity book or worksheet
- themed color-by-number pages
- numbered coloring pages with a matching color key
- printable color-by-number pages for children
- a color-by-number series with consistent characters across pages
- color-by-number pages with a custom color palette

## Workflow

1. Collect the required inputs:
   - `theme`: may be a short theme or a long prompt string
   - `colors`: optional comma-separated list of color names (defaults to standard 8-color crayon set)
   - `pages`: number of pages to generate (default 5)
   - `reference_image`: optional reference picture for style consistency
2. If needed, install local prerequisites:

```bash
bash skills/color-by-number/scripts/install.sh
```

3. Run the bundled script:

```bash
python3 skills/color-by-number/scripts/generate_color_by_number.py \
  --theme 'Jungle animals on a safari' \
  --pages 5 \
  --output-dir tmp
```

4. Confirm the outputs (two files per page plus a plan):
   - `<theme-slug>-<timestamp>-page-001-colored.png` … `page-XXX-colored.png`
   - `<theme-slug>-<timestamp>-page-001-bw.png` … `page-XXX-bw.png`
   - `<theme-slug>-<timestamp>-plan.md`

**Custom color palette for toddlers:**

```bash
python3 skills/color-by-number/scripts/generate_color_by_number.py \
  --theme 'Simple farm animals for young children' \
  --colors 'Red,Blue,Yellow,Green' \
  --pages 3 \
  --output-dir tmp
```

**Ocean theme with optimal 6-color set:**

```bash
python3 skills/color-by-number/scripts/generate_color_by_number.py \
  --theme 'Friendly fish swimming in the ocean' \
  --colors 'Light Blue,Dark Blue,Green,Yellow,Orange,Red' \
  --pages 4 \
  --output-dir tmp
```

### With a style reference image

```bash
python3 skills/color-by-number/scripts/generate_color_by_number.py \
  --theme 'Jungle animals on a safari' \
  --reference-image /absolute/path/to/reference.png \
  --pages 5 \
  --output-dir tmp
```

### With Gemini / Nano Banana for planning

```bash
python3 skills/color-by-number/scripts/generate_color_by_number.py \
  --provider gemini \
  --theme 'Fairy-tale forest with woodland animals' \
  --pages 5 \
  --output-dir tmp
```

## Replay From Markdown

If you want to iterate from a previous run, edit the generated Markdown file and rerun with:

```bash
python3 skills/color-by-number/scripts/generate_color_by_number.py \
  --replay-from-md tmp/your-theme-20260317-120000-plan.md \
  --output-dir tmp
```

Editable sections that affect the rerun:

- `Theme`
- `Colors`
- `Number of pages`
- `Reference image`
- `Provider`
- `Picture Descriptions`
- `Planner model`
- `Image model`

## Educational Benefits

The generator incorporates child development principles:

**🧠 Cognitive Development:**
- Number recognition (1-10)
- Color matching and identification
- Following sequential instructions
- Pattern recognition

**✋ Motor Skills:**
- Fine motor control through coloring
- Hand-eye coordination
- Grip strength development
- Precision and control

**🎨 Design Principles:**
- 5-10 sections maximum (optimal for attention span)
- Large regions sized for small hands
- High contrast for visual clarity
- Simple, recognizable shapes
- Age-appropriate themes (3-8 years)

**📏 Quality Standards:**
- Bold outlines for easy following
- Numbers large enough to read clearly
- Color legend on every page
- Consistent color-number mapping
- White background for contrast

## Color Key

The script embeds the color key both in the Markdown plan and in every B&W image prompt so the AI numbers regions accordingly:

| Number | Color |
|--------|-------|
| 1 | Red |
| 2 | Orange |
| … | … |

(Shown with actual colors in the plan file for each run.)

## Default Color Palette

When `--colors` is not provided the script uses the standard 8-color crayon set:

1. Red
2. Orange
3. Yellow
4. Green
5. Blue
6. Purple
7. Brown
8. Black

## What The Script Does

- Uses a text model to turn the theme and color palette into:
  - an exact picture list for each page that advances a cohesive storyline
  - detailed character descriptions that get included in all image prompts
  - a base style prompt for all images
- For each page:
  1. Generates a **full-color** illustration via AI (DALL-E) using only the specified palette colors
  2. Derives the **black-and-white numbered** version **deterministically** from the colored image:
     - Quantizes every pixel to the nearest palette color (no dithering)
     - Detects boundaries between color regions and draws black outlines
     - Places the palette-color number inside each region on a regular grid
     - Ensures every color that appears in the image is numbered at least once
- Exports each version as a separate PNG file so the final images stay locked to the exact palette colors
- Writes a Markdown plan with the color key, inputs, prompts used, and any failures
- Supports `--provider openai` and `--provider gemini` (experimental; image generation always uses OpenAI)
- Defaults to OpenAI for maximum reliability

## Requirements

- **For image generation**: `OPENAI_API_KEY` or `OPEN_AI_TOKEN` must be set (required)
- **For planning**: `GEMINI_API_KEY` or `GOOGLE_API_KEY` can be set to use Gemini for planning (optional)
- **Python packages**: `Pillow`, `numpy`, and `scipy` (installed automatically by `install.sh`)
- `scripts/install.sh` bootstraps Homebrew if needed, installs Python 3, Pillow, numpy, and scipy

## Notes

- **Two files per page**: each page produces a colored PNG and a B&W numbered PNG
- **Palette-locked color output**: the saved colored image is quantized to the exact requested palette, so it contains only flat palette colors with no extra shades
- **Deterministic B&W**: the numbered page is derived from the palette-locked colored image via palette quantization and edge detection – given the same colored image and palette, the B&W output is always identical
- **Complete numbering key**: every page includes a footer key that lists each number exactly once and covers the full palette with no missing colors
- **Color key**: embedded in the Markdown plan for easy printing alongside the activity pages
- **Character consistency**: the planner identifies and tracks main characters so they look the same across all pages
- **Parallel generation**: all colored pages are generated in parallel (up to 3 concurrent AI requests); B&W conversion runs immediately after each colored page is saved
- **Error resilience**: content-policy violations and API errors cause individual pages to be skipped; the rest continue
- Failed pages are reported at the end with specific error messages and documented in the plan file
