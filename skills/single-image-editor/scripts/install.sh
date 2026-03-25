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

ensure_python3() {
  command_exists python3 || fail "python3 is required."
}

write_env_template() {
  local env_path="${SKILL_DIR}/.env.example"
  if [[ ! -f "${env_path}" ]]; then
    fail "Missing ${env_path}"
  fi
  log "Verified ${env_path}"
}

print_next_steps() {
  cat <<EOF

Installation complete.

Requirements available:
- python3: $(command -v python3)

Next steps:
1. Export your API key:
   export OPENAI_API_KEY='your_api_key_here'
   export GEMINI_API_KEY='your_gemini_api_key_here'
2. Run the editor:
   python3 ${SKILL_DIR}/scripts/generate_single_image_edit.py --help
EOF
}

main() {
  ensure_python3
  write_env_template
  print_next_steps
}

main "$@"
