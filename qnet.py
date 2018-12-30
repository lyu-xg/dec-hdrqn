import numpy as np
import tensorflow as tf
from tfhelpers import fully_connected, convLayers

class Qnetwork:
    def __init__(self, sess, scope, input_dim, n_action, n_quant, quantile_init_w=0.01, distort_type='identity', distort_param=0.0,
                 quant_mean_loss=0, h_size=64, train_tracelen=4, implicit_quant=0, optimism=0.0,
                 learning_rate=0.001, huber_delta = 1.0, discount=0.99, magic=0, conv=0,
                 train_batch_size=32, is_target=False, **kwargs):
        self.sess = sess
        self.scope = scope
        self.input_dim, self.h_size = input_dim, h_size
        self.n_action, self.n_quant = n_action, n_quant
        self.quant_mass = tf.constant(1 / (self.n_quant + 1))
        self.learning_rate, self.discount = learning_rate, discount
        self.huber_delta = huber_delta
        self.magic, self.implicit_quant = magic, implicit_quant # * in implicit case, we fix the number of samples to be `n_quant`
        self.quant_mean_loss = quant_mean_loss
        self.train_tracelen, self.train_batch_size = train_tracelen, train_batch_size
        self.train_length = self.train_batch_size * self.train_tracelen
        self.train_range = tf.range(self.train_length)
        if self.n_quant:
            self.prob_tolerant = .5 / self.n_quant # TODO pass as param
        self.quantile_init_w = quantile_init_w
        self.conv = conv
        self.optimism = optimism
        self.distort_type, self.distort_param = distort_type, distort_param
        # assert(implicit_quant and not n_quant or not implicit_quant)


        if is_target:
            with tf.variable_scope(scope + 'target'):
                self.construct_Qnetwork()
            return
        
        with tf.variable_scope(scope + 'main'):
            self.construct_Qnetwork()
        self.target_network = Qnetwork(sess, scope, input_dim, n_action, n_quant,
            h_size=h_size, implicit_quant=implicit_quant, is_target=True, conv=conv, **kwargs)

        self.construct_training_method()
        self.construct_target_update_ops()
        


    def construct_Qnetwork(self):
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.tracelength = tf.placeholder(dtype=tf.int32, name='tracelength')
        self.epsilon = tf.placeholder(dtype=tf.float32, shape=[], name='epsilon')

        if self.conv:
            self.obs_input = tf.placeholder(shape=[None, *self.input_dim], dtype=tf.float32, name='obs_input')
            features = convLayers(tf.reshape(self.obs_input, [-1, *self.input_dim]))
        else:
            self.obs_input = tf.placeholder(shape=[None, self.input_dim], dtype=tf.float32, name='obs_input')
            features = fully_connected(self.obs_input, 32)
            features = fully_connected(features, self.h_size)

        self.lstm_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=self.h_size)
        self.lstm_state_in = self.lstm_cell.zero_state(self.batch_size, tf.float32)

        # * maybe memorize the online state inside tensorflow
        # * maybe rewrite the environment in tensorflow
        features, self.lstm_state = tf.nn.dynamic_rnn(
            cell=self.lstm_cell,
            inputs=tf.reshape(features, [self.batch_size, self.tracelength, self.h_size]),
            dtype=tf.float32,
            initial_state=self.lstm_state_in
        )

        if self.implicit_quant:
            self.tau = tf.random_uniform([self.batch_size * self.tracelength * self.n_quant, 1])
            self.tau = self.distort(self.tau)
            embedded_tau = fully_connected(self.tau, self.h_size)
            embedded_tau = tf.reshape(embedded_tau, [self.batch_size * self.tracelength, self.n_quant, self.h_size])
            # ! cosine and stuff not implemented

            # (bs, quant, h_size*2)
            # features = tf.concat([tf.map_fn(self.rep_row, features), embedded_tau], axis=2)
            features = tf.reshape(features, [self.batch_size * self.tracelength, 1, self.h_size]) * embedded_tau
            features = tf.reshape(features, [self.batch_size * self.tracelength * self.n_quant, self.h_size])

            # features = tf.concat([features, embedded_tau], axis=1)
            # features = tf.map_fn(
            #     lambda b: tf.reshape(b, [self.n_quant, self.h_size]), 
            #     embedded_tau
            # )

        layer3 = fully_connected(tf.reshape(features, [-1, self.h_size]), 32)

        # TODO dueling Q
        if self.implicit_quant: #or not self.n_quant:
            Q = fully_connected(layer3, self.n_action, activation=None, init_val=self.quantile_init_w)
            Q = tf.reshape(Q, [-1, self.n_quant, self.n_action])
            self.QLearn = self.Qout_dist = tf.map_fn(tf.transpose, Q) # (bs * tl, n_a, n_q)
            self.Qout = tf.reduce_mean(self.Qout_dist, axis=2)

            self.tau_large = tf.reshape(self.tau, [self.batch_size * self.tracelength, self.n_quant, 1]) > self.optimism
            self.tau_large = tf.cast(self.tau_large, tf.float32)


            self.optimistic_Qdist = Q * self.tau_large # (bs*tl, n_q, n_a)

            self.optimistic_Qout = tf.reduce_sum(self.optimistic_Qdist, 1) / (tf.reduce_sum(self.tau_large, 1) + 0.01)

            self.optimistic_predict = tf.argmax(self.Qout, 1, output_type=tf.int32)


        elif self.n_quant:
            self.QLearn = self.Qout_dist = tf.reshape(
                fully_connected(layer3, self.n_action * self.n_quant, activation=None, init_val=self.quantile_init_w),
                [-1, self.n_action, self.n_quant]
            )
            self.Qout = tf.reduce_mean(self.Qout_dist, axis=2)

        else:
            self.QLearn = self.Qout = fully_connected(layer3, self.n_action, activation=None)

        self.predict = tf.argmax(self.Qout, 1, output_type=tf.int32) # (batchsize*tracelen,)
        
        # tf.summary.histogram('online_actions', self.predict)

        self.online_action = (self.predict if not self.optimism else self.optimistic_predict)[-1]

    def construct_training_method(self):
        self.transition_rewards = tf.placeholder(tf.float32, shape=[None])
        self.transition_terminals = tf.placeholder(tf.float32, shape=[None])
        self.transition_actions = tf.placeholder(tf.int32, shape=[None])
        self.transition_valid = tf.placeholder(tf.float32, shape=[None])
        self.hysteretic = tf.placeholder(shape=None, dtype=tf.float32, name='hysteretic')

        R = tf.reshape(self.transition_rewards,   (-1 ,1)) if self.n_quant else self.transition_rewards
        T = tf.reshape(self.transition_terminals, (-1, 1)) if self.n_quant else self.transition_terminals

        # double-Q-in-single-pass hack
        Q_s, Q_s_next = tf.split(self.QLearn, 2)
        predict, predict_next = tf.split(self.predict, 2)
        # self.train_len_range = tf.constant(list(range(self.train_batch_size * self.train_trace_length)))

        # select action using main network, using Q values from target network
        Q       = self.select_actions(Q_s, self.transition_actions)
        targetQ = self.select_actions(self.target_network.QLearn, predict_next)

        targetQ = tf.stop_gradient(targetQ)

        # here goes BELLMAN
        targetQ = R + self.discount * targetQ * (- T + 1)

        
        '''
        We have 4 ways to calculate loss 
            1. DRQN (huber loss not useful because we have only one task, hence reward magnitude has no variance)
            2. Hysteretic DRQN
            3. Distributional DRQN
            4. Distributional Hysteretic DRQN (unknown ! okay let's figure it out)
                We can reduce the hysteretic when likelyhood of the target is low
            (Maybe we have a fifth case here, called Magic, toggled by `self.magic`)
        4 cases controlled by two parameters, `hysteretical` and `n_quant`
        '''

        if self.distributional:
            # handles case 3 and 4
            self.loss = self.quantile_loss(Q, targetQ)
        else:
            tf.summary.scalar("Q"+self.scope, Q[-1])
            tf.summary.scalar('target_Q'+self.scope, targetQ[-1])
            td_err = (targetQ - Q) * self.transition_valid
            # handles case 1 and 2, set hysteretic to 1 to effectively turn
            td_err = tf.maximum(self.hysteretic * td_err, td_err)
            self.loss = tf.reduce_mean(tf.square(td_err))

        self.update_model = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        

    def quantile_loss(self, dist, target_dist):
        tf.summary.scalar('Q'+self.scope, tf.reduce_mean(dist[-1]))
        tf.summary.scalar('target_Q'+self.scope, tf.reduce_mean(target_dist[-1]))

        tf.summary.histogram("Qdist"+self.scope, dist[-1])
        tf.summary.histogram('target_Qdist'+self.scope, target_dist[-1])

        tf.summary.histogram('Qdist_min' + self.scope, tf.reduce_min(dist, axis=1))
        tf.summary.histogram('Qdist_max' + self.scope, tf.reduce_max(dist, axis=1))

        T = tf.map_fn(lambda b: (self.rep_row(b, self.n_quant)), target_dist)
        D = tf.map_fn(lambda b: tf.transpose(self.rep_row(b, self.n_quant)), dist)
        residual = T - D
        tau_row = list((2 * np.array(list(range(self.n_quant))) + 1) / (2 * self.n_quant))
        tau = tf.constant([[tau_row for _ in range(self.n_quant)] for b in range(self.train_length)], dtype=tf.float32)

        if self.implicit_quant:
            self.train_tau, _ = tf.split(self.tau, 2)
            self.train_tau = tf.reshape(self.train_tau, [self.train_length, self.n_quant])
            tau = tf.map_fn(self.rep_row, self.train_tau)


        # loss = tf.maximum(tau * residual, (tau - 1) * residual)
        residual_counterweights = tau - tf.cast(residual > 0, tf.float32)
        loss = self.huber_loss(residual) * tf.abs(residual_counterweights)

        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1), axis=1)
        
        if self.magic:
            target_prob = self.calculate_batch_target_prob(dist, target_dist)
            target_prob = tf.stop_gradient(target_prob) # ? assume higher prob = higher hysteretic?

        is_positive_update = (tf.reduce_mean(dist, axis=1) - tf.reduce_mean(target_dist, axis=1)) > 0

        hyteresis_mask = (
            # negative update, we reduce loss by beta scale of `self.hysteretic`
            tf.cast(tf.logical_not(is_positive_update), tf.float32) * tf.maximum((target_prob if self.magic else 0.0), self.hysteretic) + # ! near zero target_prob makes Q values negatively explode
            # positive update, alpha learning rate = 1
            tf.cast(is_positive_update, tf.float32)
        )


        return tf.reduce_mean(self.transition_valid * loss * hyteresis_mask)


    def calculate_batch_target_prob(self, D, T):
        # return tf.map_fn(self.calculate_target_prob, tf.concat([D, T], axis=1))

        if self.implicit_quant:
            # sort_idx = tf.contrib.framework.argsort(self.train_tau)
            # D = tf.gather_nd(D, sort_idx)
            # tau = tf.gather(self.train_tau, sort_idx)
            D = tf.map_fn(tf.contrib.framework.sort, D)
            tau = tf.map_fn(tf.contrib.framework.sort, self.train_tau)
            delta_tau = tau[:, 1:] - tau[:, :-1]
        else:
            delta_tau = tf.ones([self.train_length, self.n_quant - 1]) / (self.n_quant - 1)

        delta_tau = tf.map_fn(self.rep_row, delta_tau)

        D_l = tf.map_fn(lambda b: self.rep_row(b[:-1], self.n_quant), D)
        D_h = tf.map_fn(lambda b: self.rep_row(b[1:],  self.n_quant), D)
        T = tf.map_fn(lambda b: tf.transpose(self.rep_row(b, self.n_quant - 1)), T)
        T_l, T_h = T - self.prob_tolerant, T + self.prob_tolerant
        # ZEROS = tf.zeros([self.train_length, self.n_quant, self.n_quant - 1])
        # * we now assume target is bad when distribution is bad
        hits = tf.where(
            D_h > D_l,
            tf.maximum((tf.minimum(D_h, T_h) - tf.maximum(D_l, T_l)), 0) / (D_h - D_l),
            tf.zeros_like(T)
        )
        hits_prob = hits * delta_tau
        prob = tf.reduce_mean(tf.reduce_sum(hits_prob, axis=1), axis=1)
        tf.summary.histogram('target_sample_likelihoods' + self.scope, prob)
        tf.summary.scalar('target_sample_likelihood' + self.scope, prob[-1])
        return prob

    def sort(t):
        return tf.gather()
        

    # def calculate_target_prob(self, dists):
    #     D, T = tf.split(dists, 2)
    #     L, U = D[:-1], D[1:]
    #     return tf.reduce_mean(tf.map_fn(lambda t: self.calculate_sample_prob(L, U, t)), T)

    # def calculate_sample_prob(self, q_h, q_l, sample):
    #     sample = tf.tile([sample], [self.n_quant - 1])
    #     l, h = sample - self.prob_tolerant, sample + self.prob_tolerant
    #     hits = tf.maximum(tf.minimum(q_h, h) - tf.maximum(q_l, l), 0)
    #     weighted_hits = hits * (q_h - q_l) # assuming linearity within quantile bins
    #     prob = tf.reduce_sum(weighted_hits) / (self.n_quant + 1)
    #     prob = tf.Print(prob, [prob], 'prob')
    #     return prob

    def distort(self, samples):
        return {
            'identity':   lambda x: x,
            'wang': self.distorter_wang,
        }[self.distort_type](samples)

    def distorter_wang(self, samples):
        bias = self.distort_param or self.epsilon # * when param is 0, use epsilon as distortion bias
        normal = tf.distributions.Normal(0.0, 0.1)
        return normal.cdf(normal.quantile(samples) + bias)

    def select_actions(self, Q, A):
        # Q: (batch_size * tracelen, a_size) OR (batch_size * tracelen, a_size, n_quantile)
        # A: (batch_size * tracelen, )
        # RETURNS: (batch_size * tracelen, ) OR (batch_size * tracelen, n_quantile)
        return tf.gather_nd(Q, tf.transpose(tf.stack([self.train_range, A])))
        
    def rep_row(self, row, times=0):
        times = times or self.n_quant
        return tf.reshape(tf.tile(row, [times]), [times,-1])

    def huber_loss(self, residual):
        K = self.huber_delta
        abs_residual = tf.abs(residual)

        if not K:
            return abs_residual

        # qradratic portion when residual is small
        small_res = 0.5 * tf.square(residual)    * tf.cast(abs_residual <= K, tf.float32)
        # linear portion when residual is bigger
        large_res = K * (abs_residual - 0.5 * K) * tf.cast(abs_residual >  K, tf.float32)
        return small_res + large_res

    def construct_target_update_ops(self):
        self.target_update_ops = []
        M = tf.trainable_variables(scope=self.scope+'main')
        T = tf.trainable_variables(scope=self.scope+'target')
        for m, t in zip(M, T):
            self.target_update_ops.append(t.assign(m.value()))
    


    ########################################################################################
    # Exposed "agent-ly" methods, used by "Team" instances to act and learn online

    def get_action_ops(self, obs, state, epsilon):
        # RETURNS: operations and feed_dicts
        ops = [[self.online_action, self.lstm_state]]
        feeds = {
            self.batch_size: 1,
            self.tracelength: 1,
            self.lstm_state_in: state,
            self.obs_input: [obs],
            self.epsilon: epsilon
        }
        return ops, feeds

    def get_training_ops(self, O, A, R, O_next, T, V, epsilon, adtl_feeds={}):
        # no need to specify starting state, starts with ZERO_STATE
        feeds = {
            self.target_network.obs_input: O_next,
            self.target_network.batch_size: self.train_batch_size,
            self.target_network.tracelength: self.train_tracelen,
            self.target_network.epsilon: epsilon,
            self.obs_input: np.concatenate([O, O_next]),
            self.batch_size: self.train_batch_size * 2,
            self.tracelength: self.train_tracelen,
            self.epsilon: epsilon,
            self.transition_actions: A,
            self.transition_rewards: R,
            self.transition_terminals: T,
            self.transition_valid: V
        }
        feeds.update(adtl_feeds)
        return [self.update_model], feeds
        
        

    ########################################################################################
    # Readability helpers

    @staticmethod
    def replace_nan_with_zero(x):
        return tf.where(tf.is_nan(x), tf.zeros_like(x), x)

    @property
    def distributional(self):
        return bool(self.n_quant)

    @property
    def ZERO_STATE(self):
        return (np.zeros((1, self.h_size)),) * 2
