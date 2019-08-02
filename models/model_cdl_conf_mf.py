import tensorflow as tf
import pickle
from models import mf_sgd
from sklearn.metrics import mean_squared_error
import numpy as np


class ConvolutionalCDL:
    def __init__(self, embed_matrix, dataset,
                 projection_dim=50, vanilla_dim=150, num_filters=50,
                 drop_ratio=0.1, epochs=50, batch_size=128, lr=0.0001,
                 lambda_q=1., lambda_v=1., lambda_w=1.,
                 out_path=None):
        # input data
        self.embed_matrix = embed_matrix
        self.texts = dataset.review_matrix
        self.rating_matrix = dataset.get_train_rating_matrix()
        self.valid_set = dataset.get_valid_rating_matrix()
        self.num_u = dataset.train_user_num()
        self.num_v = dataset.train_item_num()

        # network config
        self.n_grams = [3, 4, 5]
        self.num_filters = num_filters
        self.vanilla_dim = vanilla_dim
        self.projection_dim = projection_dim

        # dimensions
        self.max_text_len = self.texts.shape[1]
        self.embed_dim = embed_matrix.shape[1]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr

        # MF configurations
        self.lambda_q = lambda_q
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
        for gram_size in self.n_grams:
            w = tf.Variable(self.initializer([gram_size, self.embed_dim, 1, self.num_filters]), dtype=tf.float32)
            b = tf.Variable(self.initializer([self.num_filters]), dtype=tf.float32)

            x = self.conv2d(embed_input, w, b, self.embed_dim)
            x = self.maxpool2d(x, self.max_text_len)
            units.append(x)
            weights.append(w)
            biases.append(b)

        fsize = sum(i * self.num_filters for i in range(len(self.n_grams)))
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

    def training(self, verbose=True):
        print('Start training...')
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        if self.out_path:
            train_writer = tf.summary.FileWriter('%s/tf/train' % self.out_path, sess.graph)

        sess.run(tf.global_variables_initializer())

        mf = mf_sgd.SGD(dataset=self.rating_matrix, n_factors=self.projection_dim,
                        n_items=self.num_v, n_users=self.num_u,
                        lambda_q=self.lambda_q)

        val_losses = []
        for epoch in range(0, self.epochs):
            print("EPOCH %s / %s: " % (epoch + 1, self.epochs))

            v_conv_arr = []
            for i in range(0, self.texts.shape[0], self.batch_size):
                v_conv_arr.append(sess.run(self.embed_texts,
                                           feed_dict={self.texts_with_embed_words:
                                                      self.texts[i: i + self.batch_size]}))
            v_conv = np.vstack(v_conv_arr)
            if verbose:
                print('Initial embedding completed: ', v_conv.shape)

            mu, pu, qi, bu, bi = mf.run_epoch(qi_cdl=v_conv)
            err_rmse, err_mae = mf.current_error()

            val_loss = mean_squared_error(self.valid_set[:, -1], mf.predict_dataset(self.valid_set)) ** 0.5
            val_losses.append(val_loss)

            # stop early if during last 3 epochs error is only increasing
            if val_losses[-3:] and all(loss > val_losses[-3:][0] for loss in val_losses[-2:]):
                print('Stopping early because loss %s is larger than past losses %s' % (
                      val_losses[-3:][0], val_losses[-2:]))
                break

            repr_losses = []
            model_losses = []
            for i in range(0, self.texts.shape[0], self.batch_size):
                print('\n') if i % 1000 == 0 else print(".", end='')

                x_train_batch = self.texts[i: i + self.batch_size]
                y_train_batch = qi[i: i + self.batch_size]
                _, model_loss, repr_loss = sess.run([self.optimizer, self.loss, self.loss_v],
                                                    feed_dict={self.texts_with_embed_words: x_train_batch,
                                                               self.mf_v: y_train_batch})
                repr_losses.append(repr_loss)
                model_losses.append(model_loss)

            if verbose:
                print("\nSGD LOSS RMSE = %s, MAE = %s" % (err_rmse, err_mae))
                print("REPRESENTATION LOSS %s" % np.mean(repr_losses))
                print("MODEL LOSS %s" % np.mean(model_losses))
                print("VALIDATION LOSS %s" % val_loss)

            # save log files
            if self.out_path:
                # dump summaries
                summary = tf.Summary()
                summary.value.add(tag='Representation Loss', simple_value=np.mean(repr_losses))
                summary.value.add(tag='Model Loss', simple_value=np.mean(model_losses))
                summary.value.add(tag='SGD Loss', simple_value=err_rmse)
                summary.value.add(tag='Val Loss', simple_value=val_loss)
                train_writer.add_summary(summary, epoch + 1)
                # dump model and pickles
                if epoch % 5 == 0:
                    # save tensorflow model
                    self.saver.save(sess, '%s/tf/model_epoch_%s.ckpt' % (self.out_path, epoch))
                    # save matrices and biases
                    with open('%s/pickles/mx_epoch_%s.pickle' % (self.out_path, epoch), 'wb') as handle:
                        pickle.dump({'mu': mu, 'pu': pu, 'qi': qi, 'bu': bu, 'bi': bi}, handle,
                                    protocol=pickle.HIGHEST_PROTOCOL)

        sess.close()
        return mu, pu, qi, bu, bi

    @staticmethod
    def conv2d(x, W, b, embed_dim):
        x = tf.nn.conv2d(x, W, strides=[1, 1, embed_dim, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    @staticmethod
    def maxpool2d(x, k):
        return tf.nn.max_pool(x, ksize=[1, k, 1, 1], strides=[1, k, 1, 1], padding='SAME')
