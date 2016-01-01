import sys
import os
import operator
import random
from util import *
from simulator import *
from expansion import *
from retmath import *

class ActionType:
    FEEDBACK_BY_DOC=1
    FEEDBACK_BY_KEYTERM=2
    FEEDBACK_BY_REQUEST=3
    FEEDBACK_BY_TOPIC=4
    SHOW_RESULTS=5
    ACTION_NONE=6

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

# -------------------------------------------------#
# --------------- System Actions ------------------#
# -------------------------------------------------#

def feedbackByDoc(ret,ans,simulator,posprior,negprior,posdocs,negdocs,\
	poslengs,neglengs,doclengs,back,iteration,mu,delta,docmodeldir,\
	keytermlst,requestlst,topiclst,topicRanking,\
	leng=50,numwords=1000):
    params = {}
    params['ret'] = ret
    params['ans'] = ans
    doc = simulator.act(ActionType.FEEDBACK_BY_DOC,params)
    if doc!=None:
	posdocs.append(docmodeldir+IndexToDocName(doc))
	poslengs.append(doclengs[doc])
	#del ans[doc]
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
	params['docdir'] = '/home/shawnwun/ISDR-CMDP/docmodel/ref/'\
		+tokens[-2]+'/'
	#params['docdir'] = docmodeldir
	del keytermlst[0]
	# ask is relevant or not
	isRel = simulator.act(ActionType.FEEDBACK_BY_KEYTERM,params)
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
    # ask for request
    params = {}
    params['ans'] = ans
    params['request'] = requestlst
    request = simulator.act(ActionType.FEEDBACK_BY_REQUEST,params)
    posprior[request] = 1.0
    # model expansion
    posmodel = expansion(renormalize(posprior),posdocs,poslengs,back,iteration,mu,delta)
    negmodel = expansion(renormalize(negprior),negdocs,doclengs,back,iteration,mu,delta)
    return posmodel, negmodel

def feedbackByTopic(ret,ans,simulator,posprior,negprior,posdocs,\
	negdocs,poslengs,neglengs,doclengs,back,iteration,mu,delta,\
	docmodeldir,keytermlst,requestlst,topiclst,topicRanking,\
	leng=50,numwords=1000):
    params = {}
    params['ans'] = ans
    params['topic'] = topicRanking
    topicIdx = simulator.act(ActionType.FEEDBACK_BY_TOPIC,params)

    posdocs.append(pruneAndNormalize(topiclst[topicIdx],numwords))
    poslengs.append(leng)

    posmodel = expansion(renormalize(posprior),posdocs,poslengs,back,iteration,mu,delta)
    negmodel = expansion(renormalize(negprior),negdocs,doclengs,back,iteration,mu,delta)
    return posmodel, negmodel

def showResults():
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
    #'/home/shawnwun/ISDR-CMDP/docmodel/ref/CMVN/'
    #params['docdir']
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
