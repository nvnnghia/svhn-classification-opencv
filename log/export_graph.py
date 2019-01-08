# get list node names
#for ts in [n.name for n in tf.get_default_graph().as_graph_def().node]:
#    print(ts)

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import tensorflow as tf

meta_path = 'model.ckpt-62000.meta' # Your .meta file
output_node_names = ['stack']    # Output nodes

with tf.Session() as sess:

    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('.'))
    #with tf.variable_scope('Openpose'):
    #        (self.feed('Mconv7_stage6_L2',
    #                   'Mconv7_stage6_L1')
    #             .concat(3, name='concat_stage8'))
    for ts in [n.name for n in tf.get_default_graph().as_graph_def().node]:
      #if 'concat' in ts:
        print(ts)
    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('output_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_node_names)
    graph_io.write_graph(constant_graph, ".", "model1.pb", as_text=True)
