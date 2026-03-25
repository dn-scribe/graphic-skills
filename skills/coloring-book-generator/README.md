# Coloring Book Generator Examples

## Key Features

- **🚀 Fast parallel generation** after first page for style consistency  
- **🎨 Pure black and white** line art with no gray colors or shading
- **💪 Error resilient** - continues despite failed pages
- **📝 Full reproducibility** with detailed markdown plans

## Character Consistency Examples

For storylines with recurring characters, include detailed descriptions:

```bash
# Character-focused theme with specific descriptions
python3 scripts/generate_coloring_book.py \
  --theme "Captain Sam (red beard, blue sailor hat, striped shirt) and his parrot Polly (bright green, yellow beak) search for treasure on a tropical island" \
  --constraint "square pictures" \
  --constraint "all characters are smiling" \
  --pages 5 \
  --output-dir tmp
```

```bash
# Fantasy adventure with consistent characters
python3 scripts/generate_coloring_book.py \
  --theme "Wizard Ben (tall pointed hat, long gray beard, star-covered robe) and fairy friend Lily (tiny, purple dress, sparkly wand) cast spells in an enchanted forest" \
  --pages 4 \
  --output-dir tmp
```

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
  --constraint "square composition" \
  --constraint "astronaut helmets stay round in every scene" \
  --pages 4 \
  --output-dir tmp
```

## Multiple Hard Constraints

Use repeated `--constraint` flags for non-negotiable rules that should apply across planning and image generation:

```bash
python3 scripts/generate_coloring_book.py \
  --theme 'Happy Sukkot family scenes for young kids' \
  --reference-image /path/to/sukkah-reference.png \
  --constraint "square pictures" \
  --constraint "the sukkah roof is flat with a horizontal top edge" \
  --constraint "kids are smiling" \
  --pages 5 \
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
- Any repeated hard constraints used for the run, recorded in the markdown plan
- A reference PNG file from the first successful page for style consistency

**Performance**: Pages are generated in parallel after the first reference page for faster completion.

**Enhanced Character Consistency**: All images contain only pure black lines on white background with no gray colors or shading.

**Character Recommendations for Best Results**:
- Include character names and detailed physical descriptions in the theme
- Specify distinctive features (clothing, accessories, colors, size)
- Mention character relationships and personalities
- Use descriptive adjectives (friendly dragon, brave knight, curious kitten)
- The first page becomes the visual reference for character consistency

**Error Handling**: If some pages fail to generate (due to content policy violations or API errors):
- The script continues with remaining pages
- Failed pages are clearly reported at the end
- The plan file documents both successful and failed pages with error details

## API Keys

**Required for OpenAI runs:**
- `OPENAI_API_KEY` - Used for OpenAI planning and image generation

**Required for Gemini runs:**
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` - Used for Gemini planning and Gemini image generation

**Important Notes:** 
- The script defaults to OpenAI for both planning and image generation for maximum reliability
- `--provider gemini` now uses Gemini for both planning and image generation
- Gemini image generation uses the Gemini image-capable model configured by `--image-model` or the script default
