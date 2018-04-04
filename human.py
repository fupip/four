# -*- coding: utf-8 -*-
""" Human vs AI """
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet

class Human(object):
    def __init__(self):
        self.player=None
    
    def set_player_ind(self,p):
        self.player =p

    def get_action(self,game):
        try:
            mact =raw_input("input your move(player %s):"%(self.player))
            ua =mact.split(',')
            move= -1
            if len(ua)>=3:
                x,y,a=int(ua[0]),int(ua[1]),int(ua[2])
                print "x,y,a",x,y,a
                m  = game.location_to_move(x,y)
                xm = m*4+a
                actionlist=game.getactionlist()
                print "xm,actionlist",xm,actionlist
                if xm in actionlist:
                    move=xm

        except Exception as e:
            print e
            move =-1

        if move==-1:
            print "invalid move"
            move=self.get_action(game)
        print "return move",move
        return move
    def __str__(self):
        return "Human Player "+str(self.player)




                
                

