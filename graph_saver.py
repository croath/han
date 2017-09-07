import sys
import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util
import os

FLAGS = None

def main(_):
    output_node_names = "output_prob"

    session_config = tf.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction

    with tf.Session(config=session_config) as sess:
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        saver = tf.train.import_meta_graph(ckpt + '.meta')
        if ckpt:
            saver.restore(sess, ckpt)

        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        # for node in input_graph_def.node:
        #     print(node.name, node.op, node.input)

        output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        )

        with tf.gfile.GFile(os.path.join(FLAGS.model_dir, 'model.pb'), "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint_dir', type=str, default='/Users/croath/Desktop/checkpoint/', help='Directory for stroing checkpoint')
  parser.add_argument('--model_dir', type=str, help='Directory to save model')
  parser.add_argument('--gpu_fraction', type=float, help='GPU using percentage')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
