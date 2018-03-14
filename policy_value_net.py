# -*- coding: uft-8 -*-

import numpy as np
import tensorflow as tf

class PolicyValueNet(object):
    
    def __init__(self,modefile=None):
        self.width = 4
        self.height =4

        # Define tensorflow Neural Network

        self.input_states =tf.placeholder(tf.float32,shape =[None,4,self.height,self.width])
        self.input_states_reshaped =tf .reshape(self.input_states,[-1,self.height,self.width,4])

        self.conv1 =tf.layers.conv2d(inputs =self.input_states_reshaped,
                                     filters =32,kernel_size=[3,3],
                                     padding="same",activation=tf.nn.relu)

        self.conv2 =tf.layers.conv2d(inputs =self.conv1,filters =64,
                                     kernel_size=[3,3],padding="same",
                                     activation =tf.nn.relu)

        self.conv3 =tf.layers.conv2d(inputs =self.conv2,filters=128,
                                     kernel_size=[3,3],padding="same",
                                     activation=tf.nn.relu)

        self.action_conv=tf.layers.conv2d(inputs=self.conv3,filters=4,
                                          kernel_size =[1,1],padding="same",
                                          activation=tf.nn.relu)

        self.action_conv_flat =tf.reshape(self.action_conv,[-1,4*self.height*self.width])

        self.action_fc =tf.layers.dense(inputs=self.action_conv_flat,
                                        units=self.height*self.width,
                                        activation=tf.nn.log_softmax)

        self.eval_conv =tf.layers.conv2d(inputs=self.conv3,filters=2,
                                         kernel_size =[1,1],
                                         padding="same",
                                         activation=tf.nn.relu)

        self.eval_conv_flat =tf.reshape(self.eval_conv,[-1,2*self.height*self.width])

        self.eval_fc1 = tf.layers.dense(inputs =self.eval_conv_flat,units=64,activation=tf.nn.relu)

        self.eval_fc2 =tf.layers.dense(inputs =self.eval_conv_fc1,units=1,activation=tf.nn.tanh)


        self.labels =tf.placeholder (tf.float32,shape=[None,1])

        self.value_loss =tf.losses.mean_squared_error(self.labels,self.eval_fc2)

        self.mcts_probs =tf.placeholder(tf.float32,shape=[None,self.height,self.width])

        self.policy_loss =tf.negative(tf.reduce_mean(
            tf.reduce_sum(tf.multiply(self.mcts_probs,self.action_fc),1)))

        l2_penalty_beta =1e-4
        vars =tf.trainable_variables()
        l2_penalty =l2_penalty_beta*tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])

        self.loss =self.value_loss+self.policy_loss+l2_penalty

        self.learning_rate =tf.placeholder(tf.float32)
        self.optimizer =tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)

        self.session=tf.Session()

        self.entropy =tf.negative(tf.reduce_mean(
            tf.reduce_sum(tf.exp(self.action_fc)*self.action_fc,1)))

        init =tf.global_variable_initialzier()
        self.session.run(init)

        self.save=tf.train.Saver()

        if model_file is not None:
            self.restore_movel(model_file)
    def policy_value(self,state_batch):
        log_action_probs,value=self.session.run(
            [self.action_fc,self.eval_fc2],
            feed_dict={self.input_states:state_batch}
        )
        act_probs =np.exp(log_act_probs)

        return act_probs,value
    def policy_value_fn(self,game):
        legal_actions = game.actions
        current_state =np.ascontiguous(game.current_state().reshape(
            -1,4,self.width,self.height))

        act_probs,value =self.policy_value(current_state)
        act_probs =zip(legal_actions,act_probs[0][legal_actions])

        return act_probs,value

    def train_step(self,state_batch,mcts_probs,winner_batch,lr):
        winner_batch =np.reshape(winner_batch,(-1,1))

        loss,entropy,_ =self.session.run(
            [self.loss,self.entropy,self.optimizer],
            feed_dict={self.input_states:state_batch,
                       self.mcts_probs:mcts_probs,
                       self.labels:winner_batch,
                       self.learning_rate:lr})
        return loss,entropy

    def save_model(self,model_path):
        self.saver.save(self.session,model_path)

    def restore_model(self,model_path):
        self.saver.restore(self.session,model_path)

        reutrn 
