# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import time

class PolicyValueNet(object):
    
    def __init__(self,width,height,model_file=None):
        self.width = width
        self.height =height
        self.total_step =0 

        # Define tensorflow Neural Network


        # Action Net to PolicyLoss
        self.input_states =tf.placeholder(tf.float32,shape =[None,10,self.height,self.width])
        self.input_states_reshaped =tf.transpose(self.input_states,[0,2,3,1])


        self.conv1 =tf.layers.conv2d(inputs =self.input_states_reshaped,
                                     filters =32,kernel_size=[2,2],
                                     padding="same",activation=tf.nn.relu)

        self.conv2 =tf.layers.conv2d(inputs =self.conv1,filters =64,
                                     kernel_size=[2,2],padding="same",
                                     activation =tf.nn.relu)

        self.conv3 =tf.layers.conv2d(inputs =self.conv2,filters=128,
                                     kernel_size=[2,2],padding="same",
                                     activation=tf.nn.relu)

        self.action_conv=tf.layers.conv2d(inputs=self.conv3,filters=4,
                                          kernel_size =[1,1],padding="same",
                                          activation=tf.nn.relu)

        self.action_conv_flat =tf.reshape(self.action_conv,[-1,4*self.height*self.width])

        self.action_fc =tf.layers.dense(inputs=self.action_conv_flat,
                                        units=self.height*self.width*4,
                                        activation=tf.nn.log_softmax)


        self.mcts_probs =tf.placeholder(tf.float32,shape=[None,self.height*self.width*4 ])

        self.policy_loss =tf.negative(tf.reduce_mean(
            tf.reduce_sum(tf.multiply(self.mcts_probs,self.action_fc),1)))
        
        # Eval Net to ValueLoss




        self.eval_conv =tf.layers.conv2d(inputs=self.conv3,filters=2,
                                         kernel_size =[1,1],
                                         padding="same",
                                         activation=tf.nn.relu)

        self.eval_conv_flat =tf.reshape(self.eval_conv,[-1,2*self.height*self.width])

        self.eval_fc1 = tf.layers.dense(inputs =self.eval_conv_flat,units=64,activation=tf.nn.relu)

        self.eval_fc2 =tf.layers.dense(inputs =self.eval_fc1,units=1,activation=tf.nn.tanh)


        self.labels =tf.placeholder (tf.float32,shape=[None,1])

        self.value_loss =tf.losses.mean_squared_error(self.labels,self.eval_fc2)

        # L2 penalty

        l2_penalty_beta =1e-4
        vars =tf.trainable_variables()
        l2_penalty =l2_penalty_beta*tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])

        # Total Loss

        self.loss =self.value_loss+self.policy_loss+l2_penalty

        self.learning_rate =tf.placeholder(tf.float32)
        self.optimizer =tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)

        self.session=tf.Session()

        self.entropy =tf.negative(tf.reduce_mean(
            tf.reduce_sum(tf.exp(self.action_fc)*self.action_fc,1)))

        init =tf.global_variables_initializer()
        self.session.run(init)

        run_id="runid_"+str(int(time.time()))
        self.summary_writer = tf.summary.FileWriter('/tmp/tensorflowlogs/'+run_id, self.session.graph)

        tf.summary.scalar("loss",self.loss)
        tf.summary.scalar("entropy",self.entropy)

        self.mergedall = tf.summary.merge_all()

        self.saver=tf.train.Saver()

        if model_file is not None:
            self.restore_model(model_file)


    def policy_value(self,state_batch):
        log_act_probs,value=self.session.run(
            [self.action_fc,self.eval_fc2],
            feed_dict={self.input_states:state_batch}
        )
        act_probs =np.exp(log_act_probs)

        return act_probs,value

    def policy_value_fn(self,game):
        #print "--------policy_value_fn --------------"
        #raise Exception("test")
        legal_actions = game.getactionlist()
        
        #temp
        #legal_actions =[i for i in legal_actions if i<16 ]

        #print "legal_actions",legal_actions

        #print "game.stateque",len(game.stateque)
        cstate= game.current_state()

        #print "cstate",type(cstate),cstate.shape
        #print cstate

        current_state =np.ascontiguousarray(cstate.reshape(
            -1,10,self.width,self.height))


        act_probs,value =self.policy_value(current_state)

        #print "act_probs.shape",act_probs.shape
        #print act_probs
        
        temp_probs =act_probs[0]

        #print "temp_probs.shape",temp_probs.shape
        #print temp_probs
        #print temp_probs ,legal_actions

        legal_probs=temp_probs[legal_actions]

        act_probs =zip(legal_actions,legal_probs)

        return act_probs,value

    def train_step(self,state_batch,mcts_probs,winner_batch,lr):

        winner_batch =np.reshape(winner_batch,(-1,1))
        loss,entropy,opt,summary =self.session.run(
            [self.loss,self.entropy,self.optimizer,self.mergedall],
            feed_dict={self.input_states:state_batch,
                       self.mcts_probs:mcts_probs,
                       self.labels:winner_batch,
                       self.learning_rate:lr})

        self.summary_writer.add_summary(summary, self.total_step)
        self.total_step += 1




        return loss,entropy

    def save_model(self,model_path):
        self.saver.save(self.session,"./"+model_path)

    def restore_model(self,model_path):
        self.saver.restore(self.session,"./"+model_path)
        print model_path,"model restored"

        return 
