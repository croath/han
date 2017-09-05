mkdir -p /mnt/liuzhenfu/graphs_new/
mkdir -p /mnt/liuzhenfu/checkpoints_new/

nohup python3 cnn.py --data_dir=/home/liuzhenfu/training_data/positive_data/ \
 --valid_dir=/home/liuzhenfu/training_data/valid_data \
 --graph_dir=/mnt/liuzhenfu/graphs_new/ \
 --checkpoint_dir=/mnt/liuzhenfu/checkpoints_new/ \
 --charater_num=8877 \
 --epoch_num=15 \
 --batch_size=400 \
 --gpu_fraction=0.95 \
 > output.txt 2>&1&
