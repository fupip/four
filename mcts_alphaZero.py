# -*- coding: utf-8 -*-

import numpy as np
import copy


def softmax(x):
    probs = np.exp(x-np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode (object):
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):

        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        # 返回最大Q值Children Node (action)
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_sisits += 1

        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)

        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits)/(1+self._n_visits))
        return self._Q+self._u

    def is_leaf(self):
        return self._children =={}

    def is_root(self):
        return self._parent is None

class MCTS(object):
    def __init__(self,policy_value_fn, c_puct =5,n_playout =10000):
        self._root =TreeNode(None,1.0)
        self._policy =policy_value_fn
        self._c_puct =c_puct
        self._n_playout =n_playout

    def _playout(self,game):
        node =self._root
        while(1):
            if node.is_leaf():
                break
            action,node =node.select(self._c_puct)
            game.do_move(action)

        action_probs,leaf_value =self._policy(game)

        end,winner =game.win()

        if not end:
            node.expand(action_probs)
        else:
            if winner ==0 :
                leaf_value =0.0
            else:
                leaf_value =(1.0 if winner==game.get_current_player() else -1.0)
        node.update_recursive(-leaf_value)

    def get_move_probs(self,game,temp=1e-3):

        for n in range(self._n_playout):
            game_copy =copy.deepcopy(game)
            self._playout(game_copy)

        act_visits =[(act,node._n_visits) 
                     for act,node in self._root._children.items()]
        acts,visits =zip(*act_visits)
        act_probs =softmax(1.0/temp *np.log(np.array(visits) +1e-10))
        
        
        return acts,act_probs

    def update_with_move(self,last_move):
        if last_move in self._root._children:
            self._root =self._root._children[last_move]
            self._root._parent =None
        else:
            self._root =TreeNode(None,1.0)
    def __str__(self):
        return "Alpha Zero MCTS"

class MCTSPlayer(object):
    """ AI player base on AlphaZero MCTS"""
    def __init__(self,policy_value_function,c_puct=5,n_playout =2000, is _selfplay =0):
        self.mcts =MCTS(policy_value_function,c_puct,n_playout)
        self._is_selfplay =is_selfplay

    def set_player_ind(self,p):
        self.player =p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self,game,temp=1e-3,return_prob =0):
        actions =game.actions

        move_prob=np.zeros(game.width*game.height)

        if len(actions) >0:
            acts,probs =self.mcts.get_move_probs(game,temp)
            move_prob[list(acts)] =probs
            if self._is_selfplay:
                move =np.random.choice(
                    acts,
                    p=0.75*probs +0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )

                self.mcts.update_with_move(move)
            else:

                move =np.random.choice(acts,p=probs)
                self.mcts.update_with_move(-1)

            if return_prob:
                return move,move_probs
            else:
                return move
        else:
            print ("WARNING: game has no choice")

        def __str__(self):
            return "MCTS {}".format(self.player)





    


if __name__ == "__main__":
    print softmax(np.array([1, 2, 3, 4]))