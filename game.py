# -*- coding: utf-8 -*-
# author shell

import numpy as np
import os
import random
from human import Human
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet


class Game(object):
    """four game"""

    def __init__(self, **kwargs):
        self.width = 4
        self.height = 4
        self.states = {}
        self.players = [1, 2]
        self.matrix = [0]*self.width*self.height
        self.actionlist = [0]*self.width*self.height*4

        self.setvalue(0, 0, 1)
        self.setvalue(0, 1, 1)
        self.setvalue(0, 2, 1)
        self.setvalue(0, 3, 1)

        self.setvalue(3, 0, 2)
        self.setvalue(3, 1, 2)
        self.setvalue(3, 2, 2)
        self.setvalue(3, 3, 2)
        self.LastView = None
        self.CurrentView = None
        self.current_player = 1
        self.stepnum = 0
        self.simu = 0
        self.last_move = -1
        self.lastxy = ()

    def setsimu(self, value):
        self.simu = value

    def setvalue(self, x, y, v):
        m = self.location_to_move(x, y)
        self.matrix[m] = v

    def getvalue(self, x, y):
        m = self.location_to_move(x, y)
        return self.matrix[m]

    def move_to_location(self, move):
        h = move // self.width
        w = move % self.width
        return h, w

    def location_to_move(self, x, y):

        move = x * self.width + y
        if move not in range(self.width * self.height):
            return -1
        return move

    def linetostring(self, line):
        linestr = ""
        for pos in line:
            linestr = linestr+str(pos)
        return linestr

    def getrowlines(self):
        rowlines = []
        for i in range(self.width):
            line = []
            for j in range(self.height):
                line.append(self.getvalue(i, j))
            rowlines.append(line)
        return rowlines

    def getcollines(self):
        collines = []
        for i in range(self.width):
            line = []
            for j in range(self.height):
                line.append(self.getvalue(j, i))
            collines.append(line)
        return collines

    def getview(self):
        view = []
        lines = self.getrowlines()+self.getcollines()
        for line in lines:
            viewline = self.linetostring(line)
            view.append(viewline)
        return view

    def linechessnum(self, line):
        n = 0
        for x in line:
            if int(x) > 0:
                n = n+1
        return n

    def judge(self):
        self.CurrentView = self.getview()

        speclines = []
        specnum = []
        p1killview = ["2110", "0211", "1120", "0112"]
        p2killview = ["1220", "0122", "2210", "0221"]
        i = 0
        for line in self.CurrentView:
            if self.current_player == 1:
                if line in p1killview:
                    speclines.append(line)
                    specnum.append(i)
            if self.current_player == 2:
                if line in p2killview:
                    speclines.append(line)
                    specnum.append(i)
            i = i+1
        # 比较 LastView
        i = 0
        if self.LastView:
            for x in specnum:
                if self.LastView[x] != speclines[i] and self.linechessnum(self.LastView[x]) < 4:
                    self.kill(x, speclines[i])
                i = i+1

    def kill(self, x, killline):

        y = 0
        if killline in ["2110", "1220"]:
            y = 0
        if killline in ["0211", "0122"]:
            y = 1
        if killline in ["1120", "2210"]:
            y = 2
        if killline in ["0112", "0221"]:
            y = 3

        if x < 4:
            self.setvalue(x, y, 0)
            if self.simu == 0:
                print "==Player", self.current_player, "\tkill\t", str(
                    x), str(y)
                #raise Exception("kill")
        else:
            self.setvalue(y, x % 4, 0)
            if self.simu == 0:
                print "==Player", self.current_player, "\tkill\t", str(
                    x), str(y)
                #raise Exception("kill")

    def win(self):
        p1num = 0
        p2num = 0
        for x in self.matrix:
            if x == 1:
                p1num = p1num+1
            if x == 2:
                p2num = p2num+1
        #print "judge win",p1num,p2num

        actions = self.getactions()

        if len(actions) == 0:
            winner = 1

            if self.current_player == 2:
                winner = 1
            else:
                winner = 2

            return True, winner

        if p1num <= 1:
            return True, 2
        if p2num <= 1:
            return True, 1
        if self.stepnum > 100:
            return True, 0
        return False, 0

    def getactions(self):
        chess = []

        actions = []
        i = 0
        for m in self.matrix:

            if m == self.current_player:
                chess.append(i)
            i = i+1
        #print (chess)

        for c in chess:
            alist = [0, 1, 2, 3]
            x, y = self.move_to_location(c)
            #print (x, y)
            # check edge
            if x == 0:
                alist.remove(3)
            if x == 3:
                alist.remove(1)
            if y == 0:
                alist.remove(0)
            if y == 3:
                alist.remove(2)
            self.checkused(c, alist)
            for a in alist:
                actions.append((c, a))
        return actions

    def getactionlist(self):
        actions = self.getactions()
        #print ("actions",actions)
        self.actionlist = []
        for c, a in actions:
            x, y = self.move_to_location(c)
            # print(x,y,a)
            self.actionlist.append(c*4+a)

        return self.actionlist

    def moveloc(self, m, a):
        cx, cy = self.move_to_location(m)
        if a == 1:
            cx = cx+1
        if a == 2:
            cy = cy+1
        if a == 3:
            cx = cx-1
        if a == 0:
            cy = cy-1

        return cx, cy

    def action_to_xya(self, action):
        a = action % 4
        m = action//4
        x, y = self.move_to_location(m)
        return x, y, a

    def do_move(self, action):
        x, y, a = self.action_to_xya(action)
        #print "do_move_action",action,x,y,a

        self.playstep(self.current_player, x, y, a)

        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1
        self.last_move = action
        # self.show()

    def checkused(self, c, alist):

        rlist = []
        for a in alist:
            cx, cy = self.moveloc(c, a)
            v = self.getvalue(cx, cy)
            if v != 0:
                rlist.append(a)
        for rc in rlist:
            alist.remove(rc)

    def show(self):
        player1 = 1
        player2 = 2
        #print("Player", player1, "with X".rjust(3))
        #print("Player", player2, "with O".rjust(3))
        # for x in range(width):
        #   print("{0:8}".format(x), end='')
        os.system("clear")
        print('\r\n')
        for i in range(self.height-1, -1, -1):
            # print ("{0:4d}".format(i), end='')
            for j in range(self.width):
                loc = i * self.width+j
                p = self.matrix[loc]
                if p == player1:
                    print 'X'.center(8),
                elif p == player2:
                    print 'O'.center(8),
                else:
                    print '.'.center(8),
            print ('\r\n\r\n')

    def playstep(self, player, x, y, a):
        self.current_player = player
        calist = self.getactions()
        #print ("-------------------")
        #print (x,y,a,calist)
        #print ("-------------------")
        self.LastView = list(self.getview())

        m = self.location_to_move(x, y)
        if (m, a) in calist:
            cx, cy = self.moveloc(m, a)
            self.setvalue(x, y, 0)
            self.setvalue(cx, cy, player)
            self.stepnum += 1
            self.lastxy = (cx, cy)
            #print("****",self.current_player,"move",x,y,a,"***** step",self.stepnum)
            self.judge()

        else:
            print x, y, a, " is wrong", m, calist, "currentplayer", self.current_player
            raise Exception()
            return False
        return True

    def current_state(self):
        square_state = np.zeros((4, self.width, self.height))
        #print ("matrix = ",self.matrix)
        #print ("current_player=",self.current_player,self.stepnum)

        i = 0
        for m in self.matrix:

            if m != 0:
                x, y = self.move_to_location(i)
                #print x,y
                if m == self.current_player:
                    square_state[0][x, y] = 1.0
                else:
                    square_state[1][x, y] = 1.0
            i = i+1

        # Last Move
        if self.lastxy != ():
            x, y = self.lastxy
            #print x,y
            square_state[2][x, y] = 1.0

        # color
        if self.stepnum % 2 == 0:
            square_state[3] = 1.0

        #print ("square_state:",square_state)
        return square_state

    def get_current_player(self):
        return self.current_player

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        p1, p2 = self.players
        states, mcts_probs, current_players = [], [], []
        end = False
        self.current_player = 1
        player.set_player_ind(self.current_player)
        while(1):
            move, move_probs = player.get_action(
                self, temp=temp, return_prob=1)
            #print ("move,move_probs",move,move_probs)
            mcts_probs.append(move_probs)

            states.append(self.current_state())
            current_players.append(self.current_player)

            self.do_move(move)
            player.set_player_ind(self.current_player)

            if is_shown:
                self.show()
            end, winner = self.win()
            if end:
                winners_z = np.zeros(len(current_players))
                if winner != 0:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0

                player.reset_player()
                if True:
                    if winner != 0:
                        print ("Game End.Player{0} Win".format(winner))
                    else:
                        print ("Game End. Tie")
                    # print(states,mcts_probs,winners_z)

                return winner, zip(states, mcts_probs, winners_z)
        print("self_play end")

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        self.__init__()
        print "start_play", self.matrix

        self.current_player = self.players[start_player]
        p1, p2 = self.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        while(1):

            player_in_turn = players[self.current_player]

            move = player_in_turn.get_action(self)
            actionlist = self.getactionlist()
            #print "move",move,actionlist
            #print  player_in_turn,"turn"
            self.do_move(move)
            end, winner = self.win()
            if is_shown:
                print player_in_turn, " 's turn"
                self.show()

            if end:
                if winner != 0:
                    print ("Game End.Player{0} Win".format(winner))
                else:
                    print ("Game End. Tie")
                return winner

    def play(self):

        model_file = "best.model"
        best_policy = PolicyValueNet(self.width, self.height, model_file)
        mcts_player = MCTSPlayer(
            best_policy.policy_value_fn, c_puct=5, n_playout=300)
        pure_player = MCTS_Pure(c_puct=5, n_playout=500)


        human1 = Human()
        human2 = Human()
        # self.show()

        winners = []
        for i in range(10):
            winner = self.start_play(
                pure_player, mcts_player, start_player=0, is_shown=0)
            #self.start_play(pure_player,mcts_player, start_player=0, is_shown=1)
            winners.append(winner)
            print winner
        print winners


if __name__ == "__main__":
    game = Game(width=4, height=4)
    game.play()
