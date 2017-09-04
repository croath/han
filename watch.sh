head output.txt | awk '/Saving graph to:/ {print $NF}' | xargs -I{} tensorboard --logdir={}
