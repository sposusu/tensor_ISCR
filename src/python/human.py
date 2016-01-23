import sys
import os
import operator
import random
from util import *
from expansion import expansion
from retmath import *

class Simulator:
    """
      Simulates human response to retrieval machine
    """
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



class ActionType:
    FEEDBACK_BY_DOC=0
    FEEDBACK_BY_KEYTERM=1
    FEEDBACK_BY_REQUEST=2
    FEEDBACK_BY_TOPIC=3
    SHOW_RESULTS=4
    ACTION_NONE=5

def genActionSet():
    actionset = {}
    actionset[ActionType.FEEDBACK_BY_DOC] = feedbackByDoc
    actionset[ActionType.FEEDBACK_BY_KEYTERM] = feedbackByKeyterm
    actionset[ActionType.FEEDBACK_BY_REQUEST] = feedbackByRequest
    actionset[ActionType.FEEDBACK_BY_TOPIC] = feedbackByTopic
    actionset[ActionType.SHOW_RESULTS] = showResults
    return actionset

def genCostTable():
    costTable = {}
    costTable[ActionType.FEEDBACK_BY_DOC] = -30
    costTable[ActionType.FEEDBACK_BY_KEYTERM] = -10
    costTable[ActionType.FEEDBACK_BY_REQUEST] = -50
    costTable[ActionType.FEEDBACK_BY_TOPIC] = -20
    costTable[ActionType.SHOW_RESULTS] = 0
    costTable[ActionType.ACTION_NONE] = 0
    costTable['lambda'] = 1000
    return costTable

"""
  System Actions
"""

def feedbackByDoc(ret,ans,simulator,posprior,negprior,posdocs,negdocs,\
	poslengs,neglengs,doclengs,back,iteration,mu,delta,docmodeldir,\
	keytermlst,requestlst,topiclst,topicRanking,\
	leng=50,numwords=1000):

    doc = next( ( item[0] for item in ret if ans.has_key(item[0])), None )

    if doc != None:
      posdocs.append(docmodeldir+IndexToDocName(doc))
      poslengs.append(doclengs[doc])

    posmodel = expansion(renormalize(posprior),posdocs,poslengs,back,iteration,mu,delta)
    negmodel = expansion(renormalize(negprior),negdocs,doclengs,back,iteration,mu,delta)
    return posmodel, negmodel

def feedbackByKeyterm(ret,ans,simulator,posprior,negprior,posdocs,\
	negdocs,poslengs,neglengs,doclengs,back,iteration,mu,delta,\
	docmodeldir,keytermlst,requestlst,topiclst,topicRanking,\
	leng=50,numwords=1000):
    # extract keyterm
  if len(keytermlst)!=0:
  	keyterm = keytermlst[0][0]
  	params = {}
  	tokens = docmodeldir.split('/')
  	params['keyterm'] = keyterm
  	params['ans'] = ans
  	params['docdir'] = '../../ISDR-CMDP/docmodel/ref/'+tokens[-2]+'/'
    #params['docdir'] = docmodeldir
	del keytermlst[0]

  # ask is relevant or not
  docdir = '../../ISDR-CMDP/docmodel/ref/'+tokens[-2]+'/'

  ansdoc = readDocModel(docdir + IndexToDocName())

  cnt = sum( 1.0 for a in ans.iterkeys() if ansdoc.has_key(a) )

  isRel = ( True if cnt/len(ans) > 0.5 else False )

	if isRel:
	    posprior[keyterm] = 1.0
	else:
	    negprior[keyterm] = 1.0
    # model expansion
    posmodel = expansion(renormalize(posprior),posdocs,poslengs,back,iteration,mu,delta)
    negmodel = expansion(renormalize(negprior),negdocs,doclengs,back,iteration,mu,delta)
    return posmodel, negmodel

def feedbackByRequest(ret,ans,simulator,posprior,negprior,posdocs,\
	negdocs,poslengs,neglengs,doclengs,back,iteration,mu,delta,\
	docmodeldir,keytermlst,requestlst,topiclst,topicRanking,\
	leng=50,numwords=1000):
    # Ask for request
    request = requestlst[0][0]
    del requestlst[0]

    posprior[request] = 1.0
    # model expansion
    posmodel = expansion(renormalize(posprior),posdocs,poslengs,back,iteration,mu,delta)
    negmodel = expansion(renormalize(negprior),negdocs,doclengs,back,iteration,mu,delta)
    return posmodel, negmodel

def feedbackByTopic(ret,ans,simulator,posprior,negprior,posdocs,\
	negdocs,poslengs,neglengs,doclengs,back,iteration,mu,delta,\
	docmodeldir,keytermlst,requestlst,topiclst,topicRanking,\
	leng=50,numwords=1000):

    topicIdx = topicRanking[0][0]
    del topicRanking[0]

    posdocs.append(pruneAndNormalize(topiclst[topicIdx],numwords))
    poslengs.append(leng)

    posmodel = expansion(renormalize(posprior),posdocs,poslengs,back,iteration,mu,delta)
    negmodel = expansion(renormalize(negprior),negdocs,doclengs,back,iteration,mu,delta)
    return posmodel, negmodel

def showResults(ret,ans,simulator,posprior,negprior,posdocs,\
	negdocs,poslengs,neglengs,doclengs,back,iteration,mu,delta,\
	docmodeldir,keytermlst,requestlst,topiclst,topicRanking,\
	leng=50,numwords=1000):
    pass

# -------------------------------------------------#
# ------------- Simulator Actions -----------------#
# -------------------------------------------------#

def returnSelectedRequest(params):
    requestlst = params['request']
    request = requestlst[0][0]
    del requestlst[0]
    return request

def returnSelectedDoc(params):
    ret = params['ret']
    ans = params['ans']
    for item in ret:
	d = item[0]
	if ans.has_key(d):
	    return d
    return None

def returnSelectedTopic(params):
    topicRanking = params['topic']
    #index = random.randrange(0,len(topicRanking))
    index = 0
    topic = topicRanking[index][0]
    #print topic,
    del topicRanking[index]
    return topic

def returnKeytermYesNo(params):
    keyterm = params['keyterm']
    ans = params['ans']
    docdir = params['docdir']
    cnt = 0.0
    for a in ans.iterkeys():
	fname = docdir + IndexToDocName(a)
	ansdoc = readDocModel(fname)
	if ansdoc.has_key(keyterm):
	    cnt += 1.0
    if cnt/float(len(ans))>0.5:
	return True
    else:
	return False
