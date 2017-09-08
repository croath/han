import os
import tensorflow as tf
import argparse
from data_reader import read_data_sets
from data_reader import create_label_list_from_file
from chn_converter import int_to_chinese
import numpy as np

np.set_printoptions(threshold=np.inf)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    parser.add_argument('--test_dir', type=str, default=None, help='Where test images locate')
    FLAGS, unparsed = parser.parse_known_args()

    label_list = create_label_list_from_file(FLAGS.labellist)

    test_data = read_data_sets(FLAGS.test_dir, FLAGS.labellist)

    graph = load_graph(FLAGS.model_path)

    # for op in graph.get_operations():
    #     print(op.name)

    x = graph.get_tensor_by_name('import/images:0')
    y = graph.get_tensor_by_name('import/output_prob:0')
    keep_prob = graph.get_tensor_by_name('import/keep_prob:0')
    is_training = graph.get_tensor_by_name('import/is_training:0')

    with tf.Session(graph=graph) as sess:
        error_dict = {}

        while not test_data.epochs_completed:
            batch = test_data.next_batch(100, shuffle=False)
            input_images = batch[0].reshape([-1, 64, 64, 1])
            input_labels = batch[1]
            image_path = batch[2]

            y_out = sess.run(y, feed_dict={
                x: input_images,
                keep_prob: 1.0,
                is_training: False
            })

            for i in range(0, len(y_out)):
                result = y_out[i].tolist()
                max_index = result.index(max(result))

                label = input_labels[i].tolist()
                label_index = label.index(max(label))

                if label_index != max_index:
                    predict_character = int_to_chinese(label_list[max_index])
                    label_character = int_to_chinese(label_list[label_index])
                    error_dict[image_path[i]] = 'Label is ' + label_character + '. But it should be: ' + predict_character

        print("\n".join("{} - {}".format(k, v) for k, v in error_dict.items()))
