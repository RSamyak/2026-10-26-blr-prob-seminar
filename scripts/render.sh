#!/usr/bin/env bash
# Render a deck, slides.qmd by default. Quarto and Deno caches stay in dump/.
# Usage: scripts/render.sh [deck.qmd]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DECK="${1:-$ROOT/slides.qmd}"
DIR="$(cd "$(dirname "$DECK")" && pwd)"
NAME="$(basename "$DECK" .qmd)"
mkdir -p "$ROOT/dump/home" "$ROOT/dump/deno" "$ROOT/dump/cache"

export HOME="$ROOT/dump/home"
export DENO_DIR="$ROOT/dump/deno"
export XDG_CACHE_HOME="$ROOT/dump/cache"

cd "$DIR"
quarto render "$NAME.qmd"

# Pandoc writes an empty section when the body has no slides.
# KaTeX is copied beside the HTML so the TeX fonts work with no network.
KATEX="$ROOT/dump/vendor/katex-0.13.11"
if [[ ! -f "$KATEX/katex.min.js" ]]; then
  mkdir -p "$ROOT/dump/vendor"
  if curl -fsSL -o "$ROOT/dump/vendor/katex-0.13.11.tar.gz" \
    "https://github.com/KaTeX/KaTeX/releases/download/v0.13.11/katex.tar.gz"; then
    tar -xzf "$ROOT/dump/vendor/katex-0.13.11.tar.gz" -C "$ROOT/dump/vendor"
    if [[ -d "$ROOT/dump/vendor/katex" && ! -d "$KATEX" ]]; then
      mv "$ROOT/dump/vendor/katex" "$KATEX"
    fi
  fi
fi
python3 - "_output/$NAME.html" "$KATEX" <<'PY'
from pathlib import Path
import re
import shutil
import sys

html = Path(sys.argv[1])
katex = Path(sys.argv[2])
text = html.read_text()
text = re.sub(r'<section class="slide level2">\s*</section>\n?', "", text)
cdn = "https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/"
if cdn in text and (katex / "katex.min.js").is_file():
    dest = html.parent / "katex"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(katex, dest)
    text = text.replace(cdn, "katex/")
html.write_text(text)
PY
