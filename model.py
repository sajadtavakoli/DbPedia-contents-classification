import tensorflow as tf

class VGG_basedNet(object):
    def __init__(self, alphabet_size, document_max_len, num_class):
        self.alphabet_size = alphabet_size  # the number of all variant of alphabets
        self.mx_ch_stnce = document_max_len  # maximum number of character in one sentence
        self.num_cls = num_class  # number of classes
        self.embedding_size = 16  # number of features for each character
        self.filter_sizes = [3, 3, 3, 3, 3]
        self.num_filters = [64, 64, 128, 256, 512]
        self.num_blocks = [2, 2, 2, 2]
        self.learning_rate = 1e-3
        self.cnn_initializer = tf.keras.initializers.he_normal()
        self.fc_initializer = tf.truncated_normal_initializer(stddev=0.05)

        self.x = tf.placeholder(tf.int32, [None, document_max_len], name="x")
        self.y = tf.placeholder(tf.int32, [None], name="y")
        self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        self.global_step = tf.Variable(0, trainable=False)

        self.__make_model()


    def embedding_layer(self):

        with tf.name_scope("embedding"):
            init_embeddings = tf.random_uniform([self.alphabet_size, self.embedding_size], -1.0, 1.0)
            self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            x_emb = tf.nn.embedding_lookup(self.embeddings, self.x)
            self.x_expanded = tf.expand_dims(x_emb, -1)

    def conv_layers(self):

        # First Convolution Layer
        with tf.variable_scope("conv-0"):
            self.conv0 = tf.layers.conv2d(
                self.x_expanded,
                filters=self.num_filters[0],
                kernel_size=[self.filter_sizes[0], self.embedding_size],
                kernel_initializer=self.cnn_initializer,
                activation=tf.nn.relu)
            self.conv0 = tf.transpose(self.conv0, [0, 1, 3, 2])

        with tf.name_scope("conv-block-1"):
            self.conv1 = self.conv_block_likeVGG(self.conv0, 1)  # contains conv + conv + maxpooling

        with tf.name_scope("conv-block-2"):
            self.conv2 = self.conv_block_likeVGG(self.conv1, 2)  # contains conv + conv + maxpooling

        with tf.name_scope("conv-block-3"):
            self.conv3 = self.conv_block_likeVGG(self.conv2, 3)  # contains conv + conv + maxpooling

        with tf.name_scope("conv-block-4"):
            self.conv4 = self.conv_block_likeVGG(self.conv3, 4, max_pool=False)  # contains conv + conv

    def maxpool_flatten(self):

        with tf.name_scope("k-max-pooling"):
            self.h = tf.transpose(tf.squeeze(self.conv4, -1), [0, 2, 1])
            self.top_k = tf.nn.top_k(self.h, k=8, sorted=False).values
            self.h_flat = tf.reshape(self.top_k, [-1, 512 * 8])

    def dense_layers(self):

        with tf.name_scope("fc-1"):
            self.fc1 = tf.layers.dense(self.h_flat, 2048, activation=tf.nn.relu, kernel_initializer=self.fc_initializer)

        with tf.name_scope("fc-2"):
            self.fc2 = tf.layers.dense(self.fc1, 2048, activation=tf.nn.relu, kernel_initializer=self.fc_initializer)

        with tf.name_scope("fc-3"):
            self.fc3_out = tf.layers.dense(self.fc2, self.num_cls, activation=None, kernel_initializer=self.fc_initializer)
            self.predictions = tf.argmax(self.fc3_out, -1, output_type=tf.int32)

    def loss_accuracy(self):

        with tf.name_scope("loss"):
            y_one_hot = tf.one_hot(self.y, self.num_cls)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.fc3_out, labels=y_one_hot))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                     global_step=self.global_step)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def conv_block_likeVGG(self, input, i, max_pool=True):
        with tf.variable_scope("conv-block-%s" % i):
            # Two "conv-batch_norm-relu" layers.
            for j in range(2):
                with tf.variable_scope("conv-%s" % j):
                    # convolution
                    conv = tf.layers.conv2d(
                        input,
                        filters=self.num_filters[i],
                        kernel_size=[self.filter_sizes[i], self.num_filters[i-1]],
                        kernel_initializer=self.cnn_initializer,
                        activation=None)
                    # batch normalization
                    conv = tf.layers.batch_normalization(conv, training=self.is_training)
                    # relu
                    conv = tf.nn.relu(conv)
                    conv = tf.transpose(conv, [0, 1, 3, 2])

            if max_pool:
                # Max pooling
                pool = tf.layers.max_pooling2d(
                    conv,
                    pool_size=(3, 1),
                    strides=(2, 1),
                    padding="SAME")
                return pool
            else:
                return conv

    def __make_model(self):
        self.embedding_layer()
        self.conv_layers()
        self.maxpool_flatten()
        self.dense_layers()
        self.loss_accuracy()






