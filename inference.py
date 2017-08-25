import os
from log_helper import log
import config
import tensorflow as tf

class Inference(object):
    def __init__(self, model_path):
        # self.sparsity = sparsity
        # prefix = "import/"
        # self.top_k_name = prefix + "Test/Model/top_k:0"
        # self.state_in_name = prefix + "Test/Model/state:0"
        # self.input_name = prefix + "Test/Model/batched_input_word_ids:0"
        #
        # self.top_k_prediction_name = prefix + "Test/Model/top_k_prediction:1"
        # self.output_name = prefix + "Test/Model/probabilities:0"
        # self.state_out_name = prefix + "Test/Model/state_out:0"

        with open(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            print("Print graph")
            for node in graph_def.node:
                print(node.name, node.op, node.input)

            tf.import_graph_def(graph_def)

    # def predict(self, session, sentence):
    #     """Feed a sentence (str) and perform inference on this sentence """
    #     log("*" * 60)
    #     log("Sentence: " + sentence)
    #
    #     # Get word ids in sentence
    #     words = sentence.split()
    #     if words[0] not in self.word_to_id_dict:
    #         words[0] = words[0].lower()
    #
    #     sentence_ids = [self.word_to_id_dict[w] if w in self.word_to_id_dict else self.unk_id
    #                     for w in words]
    #     log(', '.join(str(e) for e in sentence_ids) + " " + str(len(sentence_ids)))
    #
    #     output_sentence_ids = sentence_ids[1:] + [self.word_to_id_dict["<eos>"]]
    #
    #     # Feed input sentence word by word.
    #     state_out = None
    #     for i in range(len(sentence_ids)):
    #         feed_values = {self.input_name: [[sentence_ids[i]]],
    #                        self.top_k_name: 5}
    #         if i > 0:
    #             feed_values[self.state_in_name] = state_out
    #         # probabilities is an ndarray of shape (batch_size * time_step) * vocab_size
    #         # For inference, batch_size = num_step = 1, thus probabilities.shape = 1 * vocab_size
    #         probabilities, top_k_predictions, state_out = session.run([self.output_name, self.top_k_prediction_name,
    #                                                                    self.state_out_name], feed_dict=feed_values)
    #         log("candidates:"+ ", ".join([self.id_to_word_dict[word_id] + '(' + str(probabilities[0][word_id]) + ')'
    #                                         for j, word_id in enumerate(top_k_predictions[0])]) +
    #               "; desired:" + self.id_to_word_dict[output_sentence_ids[i]])
