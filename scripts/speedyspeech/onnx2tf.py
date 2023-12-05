"""Converts the ONNX format graph into a frozen tensorflow graph and saves as .pb file.
"""
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np

def onnx_to_metagraph(input, model_name, quantize=False):
    ''' 
    Converts onnx graph to tensorflow metagraph
    see https://www.tensorflow.org/guide/saved_model
    '''

    # Load onnx file
    model = onnx.load("model.onnx")    

    tf_rep = prepare(model, training_mode=False, logging_level="DEBUG")

    # Input nodes to the model
    print('inputs:', tf_rep.inputs)

    # Output nodes from the model
    print('outputs:', tf_rep.outputs)
    # Export metagraph
    tf_rep.export_graph(f"{input}/{model_name}_metagraph")


def freeze_metagraph(export_dir, output_graph, model_name):
    '''
    Loads a metagraph and freezes it, saving frozen graph as a .pb
    '''
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:

        metagraph=tf.compat.v1.saved_model.loader.load(sess,['serve'],export_dir) 
        # tag here may be different ,first time you can run this code with blank tag [ ] ,error report will tell you the tag of metagraph from the path.
        sig=metagraph.signature_def['serving_default']

        input_dict=dict(sig.inputs)
        output_dict=dict(sig.outputs)

        print(input_dict) 
        graph = sess.graph
        with graph.as_default():
            output_names = [n.name for n in tf.compat.v1.get_default_graph().as_graph_def().node]

            input_graph_def = graph.as_graph_def()

            frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess, input_graph_def, output_names)
        graph = tf.compat.v1.Graph()

        with graph.as_default():
            tf.import_graph_def(frozen_graph)
            for x in list(graph.get_operations()):
                name = x.name
                print(name)
        with tf.compat.v1.gfile.GFile(f"{output_graph}/{model_name}_frozen_model.pb", "wb") as f:
            f.write(frozen_graph.SerializeToString())

        print(input_dict)
        print(output_dict)


onnx_to_metagraph('./', 'vocoder', False) # convert onxn to metagraph
freeze_metagraph(f"vocoder_metagraph", './', 'vocoder') # convert metagraph to frozen graph 
