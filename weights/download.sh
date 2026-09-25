#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p .
url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt"
if [ ! -f yolo11x.pt ]; then
  echo "downloading $url"
  curl -fL -o yolo11x.pt "$url"
else
  echo "yolo11x.pt already present"
fi