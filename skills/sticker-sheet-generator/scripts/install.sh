#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

log() {
  printf '%s\n' "$1"
}

fail() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

ensure_macos() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    fail "This installer currently supports macOS only."
  fi
}

ensure_homebrew() {
  if command_exists brew; then
    return
  fi

  log "Homebrew not found. Installing Homebrew."
  NONINTERACTIVE=1 /bin/bash -c \
    "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi

  command_exists brew || fail "Homebrew installation finished but brew is still not on PATH."
}

ensure_python3() {
  if command_exists python3; then
    return
  fi

  log "Python 3 not found. Installing python with Homebrew."
  brew install python
  command_exists python3 || fail "python3 was not found after brew install python."
}

ensure_xcode_clt() {
  if xcode-select -p >/dev/null 2>&1; then
    return
  fi

  log "Apple Command Line Tools are required for swift."
  xcode-select --install || true
  fail "Install the Apple Command Line Tools in the dialog that opened, then rerun this installer."
}

ensure_swift() {
  if command_exists swift; then
    return
  fi

  ensure_xcode_clt
  command_exists swift || fail "swift was not found after Command Line Tools installation."
}

ensure_sips() {
  command_exists sips || fail "sips is required and should be available on macOS."
}

write_env_template() {
  local env_path="${SKILL_DIR}/.env.example"
  cat >"${env_path}" <<'EOF'
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
EOF
  log "Wrote ${env_path}"
}

print_next_steps() {
  cat <<EOF

Installation complete.

Requirements available:
- python3: $(command -v python3)
- swift: $(command -v swift)
- sips: $(command -v sips)

Next steps:
1. Export your OpenAI key or Gemini key:
   export OPENAI_API_KEY='your_api_key_here'
   export GEMINI_API_KEY='your_gemini_api_key_here'
2. Run the generator:
   python3 ${SKILL_DIR}/scripts/generate_sticker_sheet.py --help
EOF
}

main() {
  ensure_macos
  ensure_homebrew
  ensure_python3
  ensure_xcode_clt
  ensure_swift
  ensure_sips
  write_env_template
  print_next_steps
}

main "$@"
