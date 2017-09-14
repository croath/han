#!/bin/sh

OP=/Users/croath/Documents/output/
mkdir -p "$OP"
python3 dump_ttf.py --output="$OP"
find "$OP" -name "*.png" | xargs -I{} convert {} -resize 64x64 +antialias -gravity center -extent 64x64 {}
