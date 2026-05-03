#!/bin/sh
set -eu

BASE_DIR="${1:-models/wd14-convnextv2.v1}"
mkdir -p "$BASE_DIR"

curl -L https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2/resolve/main/model.onnx -o "$BASE_DIR/model.onnx"
curl -L https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2/resolve/main/selected_tags.csv -o "$BASE_DIR/selected_tags.csv"
