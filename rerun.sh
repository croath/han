nohup python3 cnn.py --data_dir=/home/liuzhenfu/training_data/positive_data/ \
 --valid_dir=/home/liuzhenfu/training_data/valid_data \
 --graph_dir=/mnt/liuzhenfu/graphs/ \
 --checkpoint_dir=/mnt/liuzhenfu/checkpoints/ \
 --charater_num=8877 \
 --epoch_num=10 \
 --batch_size=400 \
 --read_from_checkpoint=True \
 --labellist=/home/liuzhenfu/han/labels.list \
 --gpu_fraction=0.5 \
 > output.txt 2>&1&
