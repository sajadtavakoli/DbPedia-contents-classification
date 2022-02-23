import tensorflow as tf
import data_preparing as dp


BATCH_SIZE = 128
WORD_MAX_LEN = 100
CHAR_MAX_LEN = 1014


x_test, y_test, alphabet_size = dp.build_char_dataset("test", CHAR_MAX_LEN)


checkpoint_file = tf.train.latest_checkpoint('VGG_basedNet')
graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        x = graph.get_operation_by_name("x").outputs[0]
        y = graph.get_operation_by_name("y").outputs[0]
        is_training = graph.get_operation_by_name("is_training").outputs[0]
        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        batches = dp.batch_iter(x_test, y_test, BATCH_SIZE, 1)
        sum_accuracy, cnt = 0, 0
        for batch_x, batch_y in batches:
            feed_dict = {
                x: batch_x,
                y: batch_y,
                is_training: False
            }

            accuracy_out = sess.run(accuracy, feed_dict=feed_dict)
            sum_accuracy += accuracy_out
            cnt += 1

        print("Test Accuracy : {0}".format(sum_accuracy / cnt))
