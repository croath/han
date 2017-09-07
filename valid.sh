nohup python3 cnn.py --data_dir=/home/liuzhenfu/training_data/positive_data/ \
 --valid_dir=/home/liuzhenfu/training_data/test_data \
 --graph_dir=/mnt/liuzhenfu/graphs_new/ \
 --checkpoint_dir=/mnt/liuzhenfu/checkpoints_new/ \
 --charater_num=8877 \
 --batch_size=400 \
 --read_from_checkpoint=True \
 --mode=test \
 --labellist=/home/liuzhenfu/han/labels.list \
 > valid.txt 2>&1&
