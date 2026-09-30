#!/bin/sh

set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
THEME="$ROOT_DIR/assets/minimal-white.css"
BUILT=0

find_browser() {
  if [ -n "${MARP_BROWSER_PATH-}" ]; then
    if [ -x "$MARP_BROWSER_PATH" ]; then
      printf '%s\n' "$MARP_BROWSER_PATH"
      return
    fi

    printf 'MARP_BROWSER_PATH is not executable: %s\n' "$MARP_BROWSER_PATH" >&2
    exit 1
  fi

  for browser_command in \
    google-chrome-stable \
    google-chrome \
    chromium \
    chromium-browser \
    microsoft-edge-stable \
    microsoft-edge \
    chrome-headless-shell
  do
    candidate=$(command -v "$browser_command" 2>/dev/null || true)
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  if [ -n "${MARP_BROWSER_CACHE-}" ]; then
    browser_cache="$MARP_BROWSER_CACHE"
  elif [ -n "${XDG_CACHE_HOME-}" ]; then
    browser_cache="$XDG_CACHE_HOME/gpu-algorithms-marp"
  elif [ -n "${HOME-}" ]; then
    browser_cache="$HOME/.cache/gpu-algorithms-marp"
  else
    printf '%s\n' \
      'Set MARP_BROWSER_CACHE or HOME so the PDF browser can be cached.' >&2
    exit 1
  fi

  browser=$(find "$browser_cache" -type f \
    \( -name 'chrome-headless-shell' -o -name 'chrome-headless-shell.exe' \) \
    -perm -111 2>/dev/null | head -n 1)

  if [ -z "$browser" ]; then
    printf '%s\n' 'Installing a headless browser for PDF export...' >&2
    npx --yes @puppeteer/browsers install chrome-headless-shell@stable \
      --path "$browser_cache" >&2

    browser=$(find "$browser_cache" -type f \
      \( -name 'chrome-headless-shell' -o -name 'chrome-headless-shell.exe' \) \
      -perm -111 2>/dev/null | head -n 1)
  fi

  if [ -z "$browser" ]; then
    printf '%s\n' 'Could not find or install a browser for PDF export.' >&2
    exit 1
  fi

  printf '%s\n' "$browser"
}

case "${1-}" in
  "")
    ;;
  clean)
    for lecture_dir in "$ROOT_DIR"/[0-9][0-9]*; do
      [ -d "$lecture_dir" ] || continue

      output_dir="$lecture_dir/out"
      if [ -d "$output_dir" ]; then
        rm -rf "$output_dir"
        printf 'Removed %s\n' "${output_dir#"$ROOT_DIR"/}"
      fi
    done
    exit 0
    ;;
  *)
    printf 'Usage: %s [clean]\n' "$0" >&2
    exit 2
    ;;
esac

if [ ! -f "$THEME" ]; then
  printf 'Missing Marp theme: %s\n' "$THEME" >&2
  exit 1
fi

BROWSER_PATH=$(find_browser)

for lecture_dir in "$ROOT_DIR"/[0-9][0-9]*; do
  [ -d "$lecture_dir" ] || continue

  for source in "$lecture_dir"/*.md; do
    [ -f "$source" ] || continue
    grep -Eq '^marp:[[:space:]]*true[[:space:]]*$' "$source" || continue

    output_dir="$lecture_dir/out"
    mkdir -p "$output_dir"
    output_base="$output_dir/$(basename "${source%.md}")"
    printf 'Building %s\n' "${source#"$ROOT_DIR"/}"

    npx --yes @marp-team/marp-cli@latest \
      --theme-set "$THEME" \
      --html \
      --output "$output_base.html" \
      "$source"

    npx --yes @marp-team/marp-cli@latest \
      --theme-set "$THEME" \
      --pdf \
      --browser chrome \
      --browser-path "$BROWSER_PATH" \
      --output "$output_base.pdf" \
      "$source"

    BUILT=$((BUILT + 1))
  done
done

if [ "$BUILT" -eq 0 ]; then
  printf '%s\n' 'No Marp presentations found in numbered directories.' >&2
  exit 1
fi

printf 'Built %s presentation(s).\n' "$BUILT"
