#!/usr/bin/env bash
set -euo pipefail

# Upload the final quantized model to a GitHub Release.
# Usage: ./scripts/upload_release.sh <tag> <file> [more files...]

tag="${1:?usage: upload_release.sh <tag> <file> [more files...]}"
shift
files=("$@")
if [[ ${#files[@]} -eq 0 ]]; then
  echo "no files to upload" >&2
  exit 2
fi

if ! gh release view "$tag" >/dev/null 2>&1; then
  gh release create "$tag" --title "$tag" --notes "Synax model $tag"
fi

gh release upload "$tag" "${files[@]}" --clobber
