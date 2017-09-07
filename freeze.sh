mkdir -p /mnt/liuzhenfu/models_new/

python3 graph_saver.py --model_dir=/mnt/liuzhenfu/models_new/ \
 --checkpoint_dir=/mnt/liuzhenfu/checkpoints_new/ \
 --gpu_fraction=0.5 \
 > freeze.txt 2>&1&
