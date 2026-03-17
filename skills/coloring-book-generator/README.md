# Coloring Book Generator Examples

## Key Features

- **🚀 Fast parallel generation** after first page for style consistency  
- **🎨 Pure black and white** line art with no gray colors or shading
- **💪 Error resilient** - continues despite failed pages
- **📝 Full reproducibility** with detailed markdown plans

## Basic Usage

Generate a 5-page underwater adventure coloring book:

```bash
python3 scripts/generate_coloring_book.py \
  --theme 'Underwater adventure with friendly sea creatures' \
  --output-dir tmp
```

## Custom Style and Pages

Generate a 3-page dinosaur coloring book with custom style:

```bash
python3 scripts/generate_coloring_book.py \
  --theme 'Friendly dinosaurs in a prehistoric forest' \
  --style 'simple cartoon style with thick outlines, perfect for young children' \
  --pages 3 \
  --output-dir tmp
```

## With Reference Image

Use a reference image to maintain consistent style:

```bash
python3 scripts/generate_coloring_book.py \
  --theme 'Space adventure with astronauts and rockets' \
  --reference-image /path/to/style-reference.jpg \
  --pages 4 \
  --output-dir tmp
```

## Using Gemini

Force use of Gemini API:

```bash
python3 scripts/generate_coloring_book.py \
  --provider gemini \
  --theme 'Magical forest with fairies and woodland animals' \
  --pages 6 \
  --output-dir tmp
```

## Replay from Previous Plan

Edit a previously generated plan and regenerate:

```bash
# First, edit the generated plan markdown file
# Then replay it:
python3 scripts/generate_coloring_book.py \
  --replay-from-md tmp/your-theme-20260317-120000-plan.md \
  --output-dir tmp
```

## Output Files

The script generates:
- Individual JPEG files for each **successfully generated** page (`theme-timestamp-page-001.jpg`, etc.)
- A markdown plan file (`theme-timestamp-plan.md`) for reproducibility
- A reference PNG file from the first successful page for style consistency

**Performance**: Pages are generated in parallel after the first reference page for faster completion.

**Pure Black & White**: All images contain only pure black lines on white background with no gray colors or shading.

**Error Handling**: If some pages fail to generate (due to content policy violations or API errors):
- The script continues with remaining pages
- Failed pages are clearly reported at the end
- The plan file documents both successful and failed pages with error details

## API Keys

**Required:**
- `OPENAI_API_KEY` - Required for reliable operation (both planning and image generation)

**Optional (Experimental):**
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` - For planning only with `--provider gemini`

**Important Notes:** 
- The script defaults to OpenAI for both planning and image generation for maximum reliability
- Gemini support is experimental and may require specific model configurations
- Even when using Gemini for planning, image generation always uses OpenAI (DALL-E)