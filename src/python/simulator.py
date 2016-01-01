import sys
import os
import operator
from util import *
from action import *


class Simulator:

    def __init__(self):
	self.answer = []
	self.policy = {}

    def setSessionAnswer(self,ans):
	self.answer = ans
	
    def addAction(self,key,func):
	self.policy[key] = func
    
    def act(self,key,params):
	return self.policy[key](params)





