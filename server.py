import os
import tensorflow as tf
import argparse

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
    FLAGS, unparsed = parser.parse_known_args()

    graph = load_graph(FLAGS.model_path)

    for op in graph.get_operations():
        print(op.name)
