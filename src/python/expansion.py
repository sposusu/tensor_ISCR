import sys
import os
import operator
from util import *

def expansion(prior,docnames,doclengs,back,iteration,mu,delta):
    
    models = []
    alphas = []

    for name in docnames:
	if isinstance(name, str):
	    models.append(readDocModel(name))
	else:
	    models.append(name)
	alphas.append(0.5)
    N = float(len(docnames))
    
    # init query model
    query = {}
    for model in models:
	for word,val in model.iteritems():
	    if query.has_key(word):
		query[word] += val/N
	    else:
		query[word] = val/N

    # EM expansion
    for it in range(iteration):
	
	# E step, Estimate P(Zw,d=1)
	aux = {}
	for m in range(len(models)):
	    model = models[m]
	    alpha = alphas[m]
	    aux[m] = {}
	    for word,val in model.iteritems():
		if query.has_key(word):
		    aux[m][word] = alpha*query[word]/\
			    (alpha*query[word]+\
			    (1-alpha)*back[word])
		else:
		    aux[m][word] = 0.0	
	# M step
	# Estimate alpha_D
	tmpmass = {}
	for m in range(len(models)):
	    alphas[m] = 0
	    model = models[m]
	    for word,val in model.iteritems():
		alphas[m] += aux[m][word]*val*doclengs[m]
		if tmpmass.has_key(word):
		    tmpmass[word] += aux[m][word]*val*doclengs[m]
		else:
		    tmpmass[word] = aux[m][word]*val*doclengs[m]
	    alphas[m] /= doclengs[m]
	
	# Estimate expanded query model
	qexpand = {}
	for word,val in prior.iteritems():
	    qexpand[word] = mu*val

	for word,val in tmpmass.iteritems():
	    if qexpand.has_key(word):
		qexpand[word] += tmpmass[word]
	    else:
		qexpand[word] = tmpmass[word]
	    
	# Normalize expanded model
	Z = 0.0
	for word,val in qexpand.iteritems():
	    Z += val
	for word in qexpand.iterkeys():
	    qexpand[word] = qexpand[word]/Z
	del query

	query = qexpand
	mu *= delta
	
	# dealloc
	del aux
	del tmpmass
	del qexpand

    del models
    del alphas
    
    qsort = sorted(query.iteritems(),key=operator.itemgetter(1),reverse=True)
    query = dict(qsort[0:100])
    #print qsort[0:3]
    #for key,val in prior.iteritems():
    #	print key, val, query[key]

    return query


def PRFExpansion(queries,ret,doclengs,background,docmodeldir,N,iteration,mu,delta):
    expanded_queries = []
    for i in range(len(queries)):
	prel = map(operator.itemgetter(0),ret[i][0:N])
	prelnames = [docmodeldir+IndexToDocName(idx) for idx in prel]
	pleng = []
	for k in prel:
	    pleng.append(doclengs[k])
	qex = expansion(queries[i],prelnames,pleng,background,iteration,mu,delta)
	expanded_queries.append(qex)
    return expanded_queries   



