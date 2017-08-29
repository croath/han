nohup python3 cnn.py --data_dir=/home/liuzhenfu/training_data/positive_data/ \
 --valid_dir=/home/liuzhenfu/training_data/negative_data \
 --graph_dir=/mnt/liuzhenfu/graphs/ \
 --checkpoint_dir=/mnt/liuzhenfu/checkpoints/ \
 --charater_num=8877 \
 --batch_size=400 \
 --read_from_checkpoint=True \
 --mode=test \
 > valid_false.txt 2>&1&
