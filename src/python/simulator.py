import sys
import os
import operator
from util import *
import action

class Simulator:
    # Will come back to this later
    def __init__(self):
        self.answer = []
        self.policy = {}

    def setSessionAnswer(self,ans):
        self.answer = ans

    def addActions(self):
        self.policy[0] = action.returnSelectedDoc
        self.policy[1] = action.returnKeytermYesNo
        self.policy[2] = action.returnSelectedRequest
        self.policy[3] = action.returnSelectedTopic

    def act(self,key,params):
        return self.policy[key](params)
