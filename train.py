# -*- coding:utf-8 -*-

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict,deque

from game import Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet


class Train():
    def __init__(self,init_model = None):
        # params of the game
        self.width  = 4
        self.height = 4
        self.game =Game()
        # params of training
        self.learn_rate= 2e-3
        self.lr_muliplier =1.0 
        self.temp =1.0
        self.n_playout =400
        self.c_puct =5
        self.buffer_size = 10000
        self.batch_size =512
        self.data_buffer =deque(maxlen =self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5
        self.kl_targ =0.02
        self.check_freq =50
        self.game_batch_num =1500
        self.best_win_ratio = 0.0

        self.pure_mcts_playout_num =1000

        if init_model:
            self.policy_value_net = PolicyValueNet(self.width,self.height,model_file= init_model)
        else:
            self.policy_value_net = PolicyValueNet(self.width,self.height)





        def run(self):
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                if len(self.data_buffer) >self.batch_size:
                    loss,entropy =self.policy_update()





