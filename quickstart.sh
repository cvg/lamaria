#!/usr/bin/env bash
set -euo pipefail

URL="https://cvg-data.inf.ethz.ch/lamaria/demo.zip"

if [ -t 1 ]; then
  PROGRESS="--progress=bar:force:noscroll"
else
  PROGRESS="--progress=dot:giga"
fi

echo "[quickstart] downloading demo.zip (~6 GB)…"
wget -c $PROGRESS -O demo.zip "$URL"

echo "[quickstart] extracting…"
unzip -q -o demo.zip
rm demo.zip

echo "[quickstart] done. demo is in ./demo"