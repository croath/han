import os
import tensorflow as tf
import argparse
from data_reader import get_real_images
from data_reader import create_label_list_from_file
from chn_converter import int_to_chinese
import numpy as np

np.set_printoptions(threshold=np.inf)
FLAGS = None

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def
        )
    return graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/Users/croath/Desktop/models/model.pb', help='Directory for stroing checkpoint')
    parser.add_argument('--labellist', type=str, default=None, help='Labels list')
    FLAGS, unparsed = parser.parse_known_args()

    label_list = create_label_list_from_file(FLAGS.labellist)

    graph = load_graph(FLAGS.model_path)

    # for op in graph.get_operations():
    #     print(op.name)

    x = graph.get_tensor_by_name('import/images:0')
    y = graph.get_tensor_by_name('import/output_prob:0')
    keep_prob = graph.get_tensor_by_name('import/keep_prob:0')

    input_images = get_real_images(['/home/liuzhenfu/training_data/test_data/AaXiHe/uni7740_着.png']).reshape([-1, 64, 64, 1])

    with tf.Session(graph=graph) as sess:

        y_out = sess.run(y, feed_dict={
            x: input_images,
            keep_prob: 1.0
        })

        print(y_out)

        chn_list = []
        prob_list = []

        for result in y_out:
            result = result.tolist()
            max_prob = max(result)
            max_index = result.index(max_prob)
            charater = int_to_chinese(label_list[max_index])

            chn_list.append(charater)
            prob_list.append(max_prob)

        print(chn_list)
        print(prob_list)
