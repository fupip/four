# -*- coding: utf-8 -*-
# author shell

from __future__ import print_function
import numpy as np
import os
import random


class Game(object):
    """four game"""

    def __init__(self, **kwargs):
        self.width = 4
        self.height = 4
        self.states = {}
        self.players = [1, 2]
        self.matrix = [0]*self.width*self.height

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
        self.current_player =1

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
                if self.LastView[x] != speclines[i]:
                    self.kill(x, speclines[i])
        

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
            print("kill "+str(x)+" "+str(y))
        else:
            self.setvalue(y, x % 4, 0)
            print ("kill "+str(y)+" "+str(x%4))

    def win(self):
        p1num =0
        p2num =0
        for x in self.matrix:
            if x ==1:
                p1num =p1num+1
            if x==2:
                p2num =p2num+1
        if p1num==1:
            return True,2
        if p2num==1:
            return True,1
        return False,0


    def getactions(self, p):
        chess = []

        actions = []
        i = 0
        for m in self.matrix:

            if m == p:
                chess.append(i)
            i = i+1
        #print (chess)

        for c in chess:
            alist = [1, 2, 3, 4]
            x,y= self.move_to_location(c)
            #print (x, y)
            # check edge
            if x == 0:
                alist.remove(3)
            if x == 3:
                alist.remove(1)
            if y == 0:
                alist.remove(4)
            if y == 3:
                alist.remove(2)
            self.checkused(c, alist)
            for a in alist:
                actions.append((c, a))
        return actions

    def moveloc(self, m, a):
        cx,cy = self.move_to_location(m)
        if a == 1:
            cx = cx+1
        if a == 2:
            cy = cy+1
        if a == 3:
            cx= cx-1
        if a == 4:
            cy = cy-1

        return cx, cy

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
        player1 =1
        player2 =2
        #print("Player", player1, "with X".rjust(3))
        #print("Player", player2, "with O".rjust(3))
#        for x in range(width):
#            print("{0:8}".format(x), end='')
        os.system("clear")
        print('\r\n')
        for i in range(self.height-1, -1, -1):
            #            print ("{0:4d}".format(i), end='')
            for j in range(self.width):
                loc = i * self.width+j
                p = self.matrix[loc]
                if p == player1:
                    print ('X'.center(8), end='')
                elif p == player2:
                    print ('O'.center(8), end='')
                else:
                    print ('.'.center(8), end='')
            print ('\r\n\r\n')

    def playstep(self, player, x, y, a):
        calist = self.getactions(player)
        self.current_player = player
        print ("-------------------")
        print (calist)
        print ("-------------------")
        self.LastView =list(self.getview())

        m = self.location_to_move(x, y)
        if (m, a) in calist:
            cx, cy = self.moveloc(m, a)
            self.setvalue(x, y, 0)
            self.setvalue(cx, cy, player)
            self.judge()
                
        else:
            print (x,y,a," is wrong")
            return False
        return True

    def autoplay(self,player):
        calist = self.getactions(player)
        calen=len(calist)
        rnd =random.randint(0,calen-1)
        (c,a)=calist[rnd]
        x,y=self.move_to_location(c)
        print ("player"+str(player)+" move "+str(x)+" "+str(y))
        return self.playstep(player,x,y,a)

    def play(self):
        game.show()
        player = 1
        while True:
            mact = raw_input("input player "+str(player)+" action:")
            ua = mact.split(',')
            if len(ua) < 3:
                continue
            result =False
            try:
                result = game.playstep(player, int(ua[0]), int(ua[1]), int(ua[2]))

            except Exception,e:
                result =False
                print (e)
                print ("input wrong")

            if result:
                game.show()
                w,p=self.win()
                if w:
                    print("Player "+str(p)+" Win!")
                    break
                else:
                    result=self.autoplay(2)
                    game.show()
                    w,p =self.win()
                    if w:
                        print("Player "+str(p)+" Win!")
                        break
            else:
                game.show()

if __name__ == "__main__":
    game = Game(width=4, height=4)
    game.play()
