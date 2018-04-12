# -*- coding:utf-8 -*-

#from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
import sys
from game import Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet


class Train():
    def __init__(self, init_model=None):
        # params of the game
        self.width = 4
        self.height = 4
        self.game = Game()
        # params of training
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = 300
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 64
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 5000
        self.best_win_ratio = 0.0

        self.pure_mcts_playout_num = 500

        if init_model:
            self.policy_value_net = PolicyValueNet(
                self.width, self.height, model_file=init_model)
        else:
            self.policy_value_net = PolicyValueNet(self.width, self.height)

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def collect_selfplay_data(self, n_games=1):

        for i in range(n_games):
            print "=====================Start===================="
            self.game = Game()
            winner, play_data = self.game.start_self_play(
                self.mcts_player, temp=self.temp)
            #print "winner",winner,play_data
            print "======================END====================="
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            #play_data = self.get_qui_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        #print "____policy___update_______"
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(
            state_batch)
        #print "old_v = ",old_v
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(
                state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs+1e-10)-np.log(new_probs+1e-10)),
                                axis=1)
                         )
            if kl > self.kl_targ * 4:
                break
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print "result-eval var=", np.var(np.array(winner_batch) - new_v.flatten(
        )), "\twinner var=", np.var(np.array(winner_batch))
        print "kl=", kl, "\tlr_mul=", self.lr_multiplier
        print "var_old : {:.3f}\tvar_new : {:.3f}".format(
            explained_var_old, explained_var_new)
        return loss, entropy

    def policy_evaluate(self, n_games=10):

        #print "_____policy__evaluation________"

        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)

        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)

        win_cnt = defaultdict(int)

        for i in range(n_games):

            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            print "winner", winner
            win_cnt[winner] += 1

            win_ratio = 1.0 * (win_cnt[1] + 0.5*win_cnt[0]) / n_games

        print "win ratio =", win_ratio
        print("num_playout:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[0]))
        return win_ratio

    def run(self, modelfile=None):
        for i in range(self.game_batch_num):
            self.collect_selfplay_data(self.play_batch_size)
            print "gamebatch :", i+1, "episode_len:", self.episode_len
            print "selfplayend,data_buffer len=", len(self.data_buffer)

            if len(self.data_buffer) > self.batch_size:
                loss, entropy = self.policy_update()
                print "loss = {:.3f}\tentropy = {:.3f}".format(loss, entropy)

            if (i+1) % self.check_freq == 0:
                print ("current self-play batch:{}".format(i+1))
                win_ratio = self.policy_evaluate()
                self.policy_value_net.save_model('current.model')
                if win_ratio > self.best_win_ratio:
                    print("new best model")
                    self.best_win_ratio = win_ratio
                    self.policy_value_net.save_model("best.model")
                    if self.best_win_ratio >= 0.8 and self.pure_mcts_playout_num < 1000:
                        print "Pure Harder"
                        self.pure_mcts_playout_num += 100
                        self.best_win_ratio = 0.0


if __name__ == '__main__':

    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    train = Train(init_model=model_path)
    train.run()
    #train.policy_evaluate()
