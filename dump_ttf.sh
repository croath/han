#!/bin/sh

OP=/Users/croath/Documents/output/
mkdir -p "$OP"
NAME=`basename $1`
mkdir -p "$OP""$NAME"
python3 dump_ttf.py --output="$OP""$NAME" --fontfile="$1"
find "$OP"/"$NAME" -name "*.png" | xargs -I{} convert {} -resize 64x64 +antialias -gravity center -extent 64x64 {}
