from tensorflow.python.framework.graph_util import convert_variables_to_constants

def export_graph(session, saved_path):
    log("Begin exporting graph!")
    graph_def = convert_variables_to_constants(session, session.graph_def,
                                               ["Test/Model/probabilities", "Test/Model/state_out",
                                                "Test/Model/top_k_prediction"])
    model_export_name = os.path.join(saved_path, 'graph.pb')
    f = open(model_export_name, "wb")
    f.write(graph_def.SerializeToString())
    f.close()
