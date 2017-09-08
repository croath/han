find $1 -type f -name ".*" -delete
find $1 -name "*.png" | xargs -I{} convert {} -resize 64x64 +antialias -gravity center -extent 64x64 {}

python3 server.py --model_path=/mnt/liuzhenfu/models_new/model.pb \
 --labellist=/home/liuzhenfu/han/labels.list \
 --test_dir=$1
