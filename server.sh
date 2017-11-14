# find $1 -type f -name ".*" -delete
# find $1 -name "*.png" | xargs -I{} convert {} -resize 64x64 +antialias -gravity center -extent 64x64 {}

python3 server.py --model_path=/Users/croath/Documents/model_opti.pb \
 --labellist=/Users/croath/Documents/labels.list \
 --test_dir=$1
