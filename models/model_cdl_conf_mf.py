import tensorflow as tf
import numpy as np
import pickle
import os


class Conv_CDL:
    def __init__(self, embed_matrix, texts, rating_matrix,
                 projection_dim=50, vanilla_dim=150, num_filters=50,
                 drop_ratio=0.1, epochs=1, batch_size=128, lr=0.0001,
                 lambda_u=1., lambda_v=1., lambda_w=1.,
                 out_path=None):
        # input data
        self.embed_matrix = embed_matrix
        self.texts = texts
        self.rating_matrix = rating_matrix

        # network config
        self.ngrams = [3, 4, 5]
        self.num_filters = num_filters
        self.vanilla_dim = vanilla_dim
        self.projection_dim = projection_dim

        # dimensions
        self.max_text_len = texts.shape[1]
        self.embed_dim = embed_matrix.shape[1]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr

        # MF configurations
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_w = lambda_w

        self.drop_ratio = drop_ratio
        self.initializer = tf.variance_scaling_initializer()
        self.build_model()
        self.out_path = out_path
        self.saver = tf.train.Saver()

    def build_model(self):
        self.texts_with_embed_words = tf.placeholder(tf.int32, shape=(None, self.max_text_len))
        self.mf_v = tf.placeholder(tf.float32, shape=(None, self.projection_dim))

        embed_input = tf.nn.embedding_lookup(self.embed_matrix, self.texts_with_embed_words)
        embed_input = tf.expand_dims(embed_input, -1)

        weights = []
        biases = []

        units = []
        for gram_size in self.ngrams:
            w = tf.Variable(self.initializer([gram_size, self.embed_dim, 1, self.num_filters]), dtype=tf.float32)
            b = tf.Variable(self.initializer([self.num_filters]), dtype=tf.float32)

            x = self.conv2d(embed_input, w, b, self.embed_dim)
            x = self.maxpool2d(x, self.max_text_len)
            units.append(x)
            weights.append(w)
            biases.append(b)

        fsize = sum(i * self.num_filters for i in range(len(self.ngrams)))
        flat = tf.concat(units, axis=3)  # flatten
        flat = tf.reshape(flat, (-1, fsize))

        w1 = tf.Variable(self.initializer([fsize, self.vanilla_dim]), dtype=tf.float32)
        b1 = tf.Variable(self.initializer([self.vanilla_dim]), dtype=tf.float32)
        weights.append(w1)
        biases.append(b1)

        dense_layer1 = tf.nn.tanh(tf.matmul(flat, w1) + b1)
        dense_layer1 = tf.nn.dropout(dense_layer1, keep_prob=1 - self.drop_ratio)

        w2 = tf.Variable(self.initializer([self.vanilla_dim, self.projection_dim]), dtype=tf.float32)
        b2 = tf.Variable(self.initializer([self.projection_dim]), dtype=tf.float32)
        weights.append(w2)
        biases.append(b2)

        self.embed_texts = tf.nn.tanh(tf.matmul(dense_layer1, w2) + b2)
        print('Shape of embed texts: ', self.embed_texts.shape)
        print('Shape of V: ', self.mf_v)
        print('Trainable params')
        print([x.shape for x in tf.trainable_variables()])

        self.Regularization = tf.reduce_sum([tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
                                             for w, b in zip(weights, biases)])
        loss_r = 1 / 2 * self.lambda_w * self.Regularization
        self.loss_v = 1 / 2 * self.lambda_v * tf.reduce_sum(tf.pow(self.embed_texts - self.mf_v, 2))
        self.loss = loss_r + self.loss_v
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def training(self):
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        if self.out_path:
            train_writer = tf.summary.FileWriter('%s/tf/train' % self.out_path, sess.graph)

        sess.run(tf.global_variables_initializer())
        mf = MF(self.rating_matrix, self.projection_dim, self.lambda_u, self.lambda_v)

        for epoch in range(0, self.epochs):
            print("EPOCH %s / %s: " % (epoch + 1, self.epochs))

            v_conv_arr = []
            for i in range(0, self.texts.shape[0], self.batch_size):
                v_conv_arr.append(sess.run(self.embed_texts,
                                           feed_dict={self.texts_with_embed_words:
                                                          self.texts[i: i + self.batch_size]}))
            V_conv = np.vstack(v_conv_arr)
            print('Initial embedding completed: ', V_conv.shape)
            U, V, beta_u, beta_v, err = mf.ALS_v3_weighted(V_conv)
            V = V.T

            auto_losses = []
            model_losses = []

            l = 0
            for i in range(0, self.texts.shape[0], self.batch_size):
                print(".", end='')
                if i % 100 == 0: print('\n')

                x_train_batch = self.texts[i: i + self.batch_size]
                y_train_batch = V[i: i + self.batch_size]

                _, model_loss, auto_loss = sess.run([self.optimizer, self.loss, self.loss_v],
                                                    feed_dict={self.texts_with_embed_words: x_train_batch,
                                                               self.mf_v: y_train_batch})
                auto_losses.append(auto_loss)
                model_losses.append(model_loss)

            print("\nALS LOSS %s" % err)
            print("REPR LOSS %s" % np.mean(auto_losses))
            print("MODEL LOSS %s" % np.mean(model_losses))

            # save log files
            if self.out_path:
                # dump summaries
                summary = tf.Summary();
                summary.value.add(tag='Autoencoder Loss', simple_value=np.mean(auto_losses))
                summary.value.add(tag='Model Loss', simple_value=np.mean(model_losses))
                summary.value.add(tag='Representation Loss', simple_value=err)
                train_writer.add_summary(summary, epoch + 1)

                # dump model and pickles
                if epoch % 5 == 0:
                    os.mkdir('%s/tf/epoch_%s/' % (self.out_path, epoch))
                    os.mkdir('%s/pickles/epoch_%s/' % (self.out_path, epoch))

                    # save tensorflow model
                    self.saver.save(sess, '%s/tf/epoch_%s/model_.ckpt' % (self.out_path, epoch))

                    # save U and V matricies from ALS
                    with open('%s/pickles/epoch_%s/U.pickle' % (self.out_path, epoch), 'wb') as handle:
                        pickle.dump(U, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open('%s/pickles/epoch_%s/V.pickle' % (self.out_path, epoch), 'wb') as handle:
                        pickle.dump(V, handle, protocol=pickle.HIGHEST_PROTOCOL)

        sess.close()
        return U, V, beta_u, beta_v

    @staticmethod
    def conv2d(x, W, b, embed_dim):
        x = tf.nn.conv2d(x, W, strides=[1, 1, embed_dim, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    @staticmethod
    def maxpool2d(x, k):
        return tf.nn.max_pool(x, ksize=[1, k, 1, 1], strides=[1, k, 1, 1], padding='SAME')
