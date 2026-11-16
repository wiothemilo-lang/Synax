#!/usr/bin/env bash
set -euo pipefail

# Download datasets/checkpoints without committing them to git.
#
# Usage:
#   ./scripts/download_data.sh gdrive <file_id> <output_path>
#   ./scripts/download_data.sh kaggle <owner/dataset> <output_dir>
#   ./scripts/download_data.sh rclone <remote:path> <output_path>

mode="${1:?usage: download_data.sh <gdrive|kaggle|rclone> ...}"
shift

case "$mode" in
  gdrive)
    file_id="${1:?missing Google Drive file id}"
    output="${2:?missing output path}"
    mkdir -p "$(dirname "$output")"
    python3 -m pip install --quiet gdown
    gdown --id "$file_id" --output "$output"
    ;;
  kaggle)
    dataset="${1:?missing owner/dataset}"
    output="${2:?missing output dir}"
    mkdir -p "$output"
    kaggle datasets download -d "$dataset" -p "$output" --unzip
    ;;
  rclone)
    source="${1:?missing remote:path}"
    output="${2:?missing output path}"
    mkdir -p "$(dirname "$output")"
    rclone copy "$source" "$(dirname "$output")" --progress
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac
