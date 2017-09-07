nohup python3 cnn.py --data_dir=/home/liuzhenfu/training_data/positive_data/ \
 --valid_dir=/home/liuzhenfu/training_data/negative_data \
 --graph_dir=/mnt/liuzhenfu/graphs_new/ \
 --checkpoint_dir=/mnt/liuzhenfu/checkpoints_new/ \
 --charater_num=8877 \
 --batch_size=386 \
 --read_from_checkpoint=True \
 --mode=test \
 --labellist=/home/liuzhenfu/han/labels.list \
 > valid_false.txt 2>&1&
