import tensorflow as tf
from sklearn.metrics import mean_squared_error
from models import mf_sgd
import pickle
import numpy as np


class CDL:
    def __init__(self, dataset, out_path=None, k=10, epochs=50, batch_size=64, lr=0.0001,
                 hidden_size=25, drop_ratio=0.1, lambda_w=1, lambda_n=1,
                 lambda_v=1, lambda_q=0.01):
        self.out_path = out_path
        self.k = k

        self.item_information_matrix = dataset.review_matrix
        self.item_information_matrix_noise = dataset.noised_review_matrix

        self.rating_matrix = dataset.get_train_rating_matrix()
        self.valid_set = dataset.get_valid_rating_matrix()

        self.n_input = self.item_information_matrix.shape[1]  # dimensionality of text representations - 1000
        self.n_hidden1 = hidden_size
        self.n_hidden2 = self.k

        self.lambda_w = lambda_w
        self.lambda_n = lambda_n
        self.lambda_v = lambda_v
        self.lambda_q = lambda_q

        self.drop_ratio = drop_ratio
        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.num_u = dataset.train_user_num()
        self.num_v = dataset.train_item_num()

        initializer = tf.variance_scaling_initializer()

        self.Weights = {
            'w1': tf.Variable(initializer([self.n_input, self.n_hidden1]), dtype=tf.float32),
            'w2': tf.Variable(initializer([self.n_hidden1, self.n_hidden2]), dtype=tf.float32),
            'w3': tf.Variable(initializer([self.n_hidden2, self.n_hidden1]), dtype=tf.float32),
            'w4': tf.Variable(initializer([self.n_hidden1, self.n_input]), dtype=tf.float32)
        }
        self.Biases = {
            # 'b1' : tf.Variable(tf.random_normal( [self.n_hidden1] , mean=0.0, stddev=1 / self.lambda_w )),
            # 'b2' : tf.Variable(tf.random_normal( [self.n_hidden2] , mean=0.0, stddev=1 / self.lambda_w )),
            # 'b3' : tf.Variable(tf.random_normal( [self.n_hidden1] , mean=0.0, stddev=1 / self.lambda_w )),
            # 'b4' : tf.Variable(tf.random_normal( [self.n_input] , mean=0.0, stddev=1 / self.lambda_w ))
            'b1': tf.Variable(tf.zeros(self.n_hidden1)),
            'b2': tf.Variable(tf.zeros(self.n_hidden2)),
            'b3': tf.Variable(tf.zeros(self.n_hidden1)),
            'b4': tf.Variable(tf.zeros(self.n_input))
        }

        self.build_model()
        self.saver = tf.train.Saver()

    def encoder(self, x, drop_ratio):
        w1 = self.Weights['w1']
        b1 = self.Biases['b1']
        l1 = tf.nn.relu(tf.matmul(x, w1) + b1)
        l1 = tf.nn.dropout(l1, keep_prob=1 - drop_ratio)

        w2 = self.Weights['w2']
        b2 = self.Biases['b2']
        l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
        l2 = tf.nn.dropout(l2, keep_prob=1 - drop_ratio)
        return l2

    def decoder(self, x, drop_ratio):
        w3 = self.Weights['w3']
        b3 = self.Biases['b3']
        l3 = tf.nn.relu(tf.matmul(x, w3) + b3)
        l3 = tf.nn.dropout(l3, keep_prob=1 - drop_ratio)

        w4 = self.Weights['w4']
        b4 = self.Biases['b4']
        l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)
        l4 = tf.nn.dropout(l4, keep_prob=1 - drop_ratio)
        return l4

    def build_model(self):
        self.model_x_0 = tf.placeholder(tf.float32, shape=(None, self.n_input))
        self.model_X_c = tf.placeholder(tf.float32, shape=(None, self.n_input))

        self.model_V = tf.placeholder(tf.float32, shape=(None, self.k))
        self.model_drop_ratio = tf.placeholder(tf.float32)

        self.V_sdae = self.encoder(self.model_x_0, self.model_drop_ratio)
        self.y_pred = self.decoder(self.V_sdae, self.model_drop_ratio)

        self.regularization = tf.reduce_sum([tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
                                             for w, b in zip(self.Weights.values(), self.Biases.values())])
        loss_r = 1 / 2 * self.lambda_w * self.regularization
        self.loss_a = 1 / 2 * self.lambda_n * tf.reduce_sum(tf.pow(self.model_X_c - self.y_pred, 2))
        loss_v = 1 / 2 * self.lambda_v * tf.reduce_sum(tf.pow(self.model_V - self.V_sdae, 2))

        self.Loss = loss_r + self.loss_a + loss_v
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.Loss)

    def training(self, verbose=True):
        print('Start training...')
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        if self.out_path is not None:
            train_writer = tf.summary.FileWriter('%s/tf/train' % self.out_path, sess.graph)

        val_losses = []

        sess.run(tf.global_variables_initializer())
        mf = mf_sgd.SGD(dataset=self.rating_matrix, n_factors=self.k,
                        n_items=self.num_v, n_users=self.num_u,
                        lambda_q=self.lambda_q)

        for epoch in range(0, self.epochs):
            print("EPOCH %s / %s" % (epoch + 1, self.epochs))

            v_sdae = sess.run(self.V_sdae, feed_dict={self.model_x_0: self.item_information_matrix_noise,
                                                      self.model_drop_ratio: self.drop_ratio})
            # calc and print ALS loss every N epochs
            mu, pu, qi, bu, bi = mf.run_epoch(qi_cdl=v_sdae)
            err_rmse, err_mae = mf.current_error()

            val_loss = mean_squared_error(self.valid_set[:, -1], mf.predict_dataset(self.valid_set)) ** 0.5
            val_losses.append(val_loss)

            # stop early if during last 3 epochs error is only increasing
            if val_losses[-3:] and all(loss > val_losses[-3:][0] for loss in val_losses[-2:]):
                print('Stopping early because loss %s is larger than past losses %s' % (val_losses[-3:][0], val_losses[-2:]))
                break

            auto_losses = []
            model_losses = []
            for i in range(0, self.item_information_matrix.shape[0], self.batch_size):
                x_train_batch = self.item_information_matrix_noise[i: i + self.batch_size]
                y_train_batch = self.item_information_matrix[i: i + self.batch_size]
                v_batch = qi[i: i + self.batch_size]

                _, my_loss, auto_loss = sess.run([self.optimizer, self.Loss, self.loss_a],
                                                 feed_dict={self.model_x_0: x_train_batch,
                                                            self.model_X_c: y_train_batch,
                                                            self.model_V: v_batch,
                                                            self.model_drop_ratio: self.drop_ratio})
                auto_losses.append(auto_loss)
                model_losses.append(my_loss)

            if verbose:
                print("SGD LOSS RMSE = %s, MAE = %s" % (err_rmse, err_mae))
                print("MODEL LOSS %s" % np.mean(model_losses))
                print("AUTOENCODER LOSS %s" % np.mean(auto_losses))
                print("VALIDATION LOSS %s" % val_loss)

            # save log files
            if self.out_path is not None:
                # dump summaries
                summary = tf.Summary()
                summary.value.add(tag='Autoencoder Loss', simple_value=np.mean(auto_losses))
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

