mkdir -p /mnt/liuzhenfu/graphs/
mkdir -p /mnt/liuzhenfu/checkpoints/

nohup python cnn.py --data_dir=/home/liuzhenfu/training_data/positive_data/ \
 --valid_dir=/home/liuzhenfu/training_data/valid_data
 --graph_dir=/mnt/liuzhenfu/graphs/ \
 --checkpoint_dir=/mnt/liuzhenfu/checkpoints/ \
 --charater_num=8877 \
 --epoch_num=10 \
 --batch_size=400 \
 > output.txt 2>&1&
