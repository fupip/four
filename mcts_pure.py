# -*- coding: utf-8 -*-

import numpy as np
import copy
from operator import itemgetter

def rollout_policy_fn(game):
    #rollout randomly
    action_probs =np.random.rand(len(game.getactionlist()))
    #随机  走一步
    return zip(game.getactionlist(),action_probs)

def policy_value_fn(game):
    action_probs =np.ones(len(game.getactionlist()))/len(game.getactionlist())
    return zip(game.getactionlist(),action_probs),0

class TreeNode(object):
    def __init__(self,parent,prior_p):
        self._parent =parent
        self._children ={}
        self._n_visits =0
        self._Q =0
        self._u =0
        self._P =prior_p

    def expand(self,action_priors):
        for action,prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self,prob)

    def select(self,c_puct):
        return max(self._children.items(),key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self,leaf_value):

        self._n_visits += 1
        self._Q += 1.0*(leaf_value -self._Q) /self._n_visits

    def update_recursive(self,leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self,c_puct):

        self._u =(c_puct *self._P*
                  np.sqrt(self._parent._n_visits)/(1+self._n_visits))

        return self._Q + self._u
    
    def is_leaf(self):
        return self._children =={}

    def is_root(self):
        return self._parent is None

class MCTS(object):
    """ Monte Carlo Tree Search """
    def __init__(self,policy_value_fn,c_puct =5,n_playout =10000):
        self._root =TreeNode(None,1.0) # prior_P=1.0
        self._policy =policy_value_fn
        self._c_puct =c_puct  # search depth
        self._n_playout = n_playout

    def _playout(self,game):

        node = self._root
        while(1):
            if node.is_leaf():
                break
            action,node =node.select(self._c_puct)
            #print "pureaction",action
            game.do_move(action)
        action_probs,_ =self._policy(game)
        end,winner = game.win()
        if not end:
            node.expand(action_probs)

        leaf_value =self._evaluate_rollout(game)
        node.update_recursive (-leaf_value)

    def _evaluate_rollout(self,game,limit=1000):
        player = game.get_current_player()
        for i in range(limit):
            end,winner =game.win()
            if end:
                break
            action_probs =rollout_policy_fn(game)
            max_action =max(action_probs,key=itemgetter(1))[0]
            #随机走一步
            game.do_move(max_action)
        else:
            #尝试结束
            print ("WARNING: rollout reached move limit")

        if winner ==0:
            return 0 #平
        else:
            return 1 if winner ==player else -1

    def get_move(self,game):
        for n in range(self._n_playout):
            game_copy =copy.deepcopy(game)
            game_copy.setsimu(1)
            self._playout(game_copy)
        # 返回访问数最多的动作
        return max(self._root._children.items(),
                   key= lambda act_node:act_node[1]._n_visits)[0]

    def update_with_move(self,last_move):

        if last_move in self._root._children:
            self._root =self._root._children[last_move]
            self._root._parent =None
        else:
            self._root =TreeNode(None,1.0)

    def __str__(self):
        return "MCTS"

class MCTSPlayer(object):
    """AI player base on MCTS"""

    def __init__(self,c_puct =5 ,n_playout=2000):
        self.mcts = MCTS(policy_value_fn,c_puct,n_playout)

    def set_player_ind(self,p):
        self.player =p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self,game):
        actions =game.getactionlist()

        if len(actions) > 0:
            move =self.mcts.get_move(game)
            self.mcts.update_with_move(-1)
            print "pure    ",self.player,"\tmove\t",move
            return move
        else:
            print("WARNING: no choice")
    def __str__(self):
        return "[MCTS Pure {}]".format(self.player)

if __name__ == "__main__":
    test=np.random.rand(10)
    test =np.ones(10)/10
    print test
