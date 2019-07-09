import keras
import keras.backend as K

import numpy as np

import tensorflow as tf

from keras.models import Model

from tqdm import trange

from VennABERS import getFVal

class WhiteboxAttack:
    def __init__(self, ivap, model, x_calib, y_calib, batch_size=128, lr=.01):
        # store parameters
        self.ivap = ivap
        self.x_calib = x_calib
        self.y_calib = y_calib
        self.batch_size = batch_size
        
        # get calibration scores
        self.logits = Model(inputs=model.input, outputs=model.layers[-2].output)
        self.calib_scores = self.logits.predict(x_calib)[:, 1]
        
        # construct optimization
        sess = K.get_session()
        with tf.variable_scope('whitebox', reuse=tf.AUTO_REUSE):
            self.eta = tf.placeholder(tf.float32)
            self.lams = tf.placeholder(tf.float32)
            self.x_origs = tf.placeholder(tf.float32)
            self.x_tildes = tf.get_variable('x_tilde', shape=[self.batch_size, *x_calib.shape[1:]])
            self.target_scores = tf.placeholder(tf.float32)
            
            logits_tensor = self.logits(self.x_tildes)
            loss = tf.reduce_mean(
                self.lams * tf.maximum(0., tf.reduce_max(abs(self.x_tildes - self.x_origs), axis=[1, 2, 3]) - self.eta) \
                + abs(logits_tensor[:, 1] - self.target_scores))
            self.opt_op = tf.train.AdamOptimizer(lr).minimize(loss, var_list=[self.x_tildes])

            init_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='whitebox')
            self.init_op = [tf.variables_initializer(init_vars),
                           tf.assign(self.x_tildes, self.x_origs)]
            self.clip_op = tf.assign(self.x_tildes, tf.clip_by_value(self.x_tildes, 0, 1))

    def batch_attack(self, x_origs, y_origs, eta, its=10, tol=1e-2):
        # sanity check
        assert x_origs.shape[0] == self.batch_size, 'Batches must be {} samples each'.format(self.batch_size)
        
        # get original scores
        orig_scores = self.logits.predict(x_origs, batch_size=self.batch_size)[:, 1]
        
        # find appropriate target vectors
        target_scores = np.copy(orig_scores)
        best_dists = np.ones(orig_scores.shape[0]) * np.inf
        for idx, (x_orig, y_orig, orig_score) in enumerate(zip(x_origs, y_origs, orig_scores)):
            for x, y, s in zip(self.x_calib, self.y_calib, self.calib_scores):
                dist = abs(orig_score - s)
                if dist < best_dists[idx] and y.argmax() != y_orig.argmax():
                    target_scores[idx] = np.copy(s)
                    best_dists[idx] = dist

        # optimize lambdas
        lowers, uppers = np.zeros(self.batch_size), np.ones(self.batch_size)
        x_sols = np.copy(x_origs)
        while (uppers - lowers).max() > tol:
            lams = lowers + (uppers - lowers)/2
            mask = np.zeros(self.batch_size).astype(np.bool)

            # start optimization
            sess = K.get_session()
            x_inits = np.clip(x_origs + 1e-4 * np.random.normal(0, 1, size=x_origs.shape), 0, 1)
            sess.run(self.init_op, feed_dict={self.x_origs: x_inits})

            # optimization loop
            for it in range(its):
                # optimization step
                sess.run(self.opt_op, feed_dict={self.lams: lams,
                                            self.x_origs: x_origs,
                                            self.target_scores: target_scores,
                                            self.eta: eta})
                sess.run(self.clip_op)

                # check intermediate solutions
                x_tildes_raw = sess.run(self.x_tildes)
                y_tildes = self.ivap.batch_predictions(x_tildes_raw, batch_size=self.batch_size)
                for idx in range(self.batch_size):
                    y_tilde = y_tildes[idx]
                    if y_tilde[-1] == 0 and abs(x_tildes_raw[idx] - x_origs[idx]).max() <= eta and y_origs[idx].argmax() != y_tilde[:2].argmax():
                        x_sols[idx] = np.copy(x_tildes_raw[idx])
                        mask[idx] = True

            # update lambdas
            lowers[mask] = lams[mask]
            uppers[np.logical_not(mask)] = lams[np.logical_not(mask)]

        return x_sols
    
    def attack(self, x_origs, y_origs, eta, its=10, tol=1e-2, verbose=True):
        assert x_origs.shape[0] % self.batch_size == 0, 'Sample size must be a multiple of batch size'
        
        num_batches = x_origs.shape[0] // self.batch_size
        x_advs = np.zeros(x_origs.shape)
        t = range(num_batches) if not verbose else trange(num_batches)
        count = 0
        for idx in t:
            start = idx * self.batch_size
            end = (idx+1) * self.batch_size
            x_advs[start:end] = self.batch_attack(x_origs[start:end], y_origs[start:end], eta, its, tol)
            
            if verbose:
                count += (abs(x_advs[start:end] - x_origs[start:end]).max(axis=(1, 2, 3)) > 0).sum()
                t.set_description('adversarials: {}'.format(count))
        return x_advs
