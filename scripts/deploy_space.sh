#!/usr/bin/env bash
# Deploy the committed HEAD to a Hugging Face Docker Space.
#
#   HF_TOKEN=hf_xxx scripts/deploy_space.sh <hf_username>/<space_name>
#
# The Space receives exactly what is committed in HEAD (nothing from your working tree, so .env
# can never leak). A Space needs YAML front matter in README.md, which would look odd on GitHub,
# so it is added only to the copy that is pushed. Secrets (ANTHROPIC_API_KEY, API_KEYS, ...) are
# set in the Space settings, never in the repo. See docs/deploy.md.
set -euo pipefail

SPACE="${1:?usage: HF_TOKEN=... $0 <hf_username>/<space_name>}"
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face access token with write permission}"
HF_USER="${SPACE%%/*}"

ROOT="$(git rev-parse --show-toplevel)"
if [ -n "$(git -C "$ROOT" status --porcelain)" ]; then
  echo "Uncommitted changes found. Commit first: the Space gets exactly what HEAD contains." >&2
  exit 1
fi

WORK="$(mktemp -d)"
trap '[ -n "${WORK:-}" ] && [ -d "$WORK" ] && rm -rf "$WORK"' EXIT

git -C "$ROOT" archive HEAD | tar -x -C "$WORK"

{
  cat <<'FRONT'
---
title: RBI Regulation RAG
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Grounded Q&A with page citations over RBI regulations
---

FRONT
  cat "$WORK/README.md"
} > "$WORK/README.space.md"
mv "$WORK/README.space.md" "$WORK/README.md"

cd "$WORK"
git init -q -b main
git add -A
git -c user.name="deploy" -c user.email="deploy@localhost" commit -q -m "Deploy $(git -C "$ROOT" rev-parse --short HEAD)"

REMOTE="${HF_REMOTE_URL:-https://huggingface.co/spaces/$SPACE}"  # override only for testing
AUTH="$(printf '%s:%s' "$HF_USER" "$HF_TOKEN" | base64 | tr -d '\n')"
git -c http.extraHeader="Authorization: Basic $AUTH" push --force "$REMOTE" HEAD:main

echo
echo "Pushed. Watch the build at: https://huggingface.co/spaces/$SPACE"
echo "Live URL: https://${SPACE%%/*}-${SPACE##*/}.hf.space  (lowercase, underscores become dashes)"
