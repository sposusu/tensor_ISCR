import math
import operator

def cross_entropy(p1,p2):
  return ( 0 if p2 == 0 else -1 * p1 * math.log(p1/p2))

def renormalize(distdict):
  Z = sum(distdict.values())
  if Z == 1:
    return distdict
  nol = {}
  for key in distdict.iterkeys():
    nol[key] = distdict[key]/Z
  return nol

def gaussian(mean,var,x):
  return 1/math.sqrt(2*math.pi*var)*\
        math.exp(-1/2*math.pow((x-mean)/math.sqrt(var),2))

def entropy(model):
  ent = 0.0
  for key,p in model.iteritems():
    ent += -1*p*math.log(p)
  return ent

def cross_entropies(model1,model2):
  cent = 0.0
  for key1,p1 in model1.iteritems():
    if model2.has_key(key1):
      p2 = model2[key1]
      cent += cross_entropy(p1,p2)
  return cent

def IDFscore(model,inv_index):
  x = sorted(model.iteritems(),key=operator.itemgetter(1),reverse=True)
  if len(x)>20:
    x = x[:20]
  scores = []
  for key, val in x:
    if len(inv_index[key])>0:
      scores.append(math.log(5047.0/float(len(inv_index[key]))))

  maxS = 0.0
  avgS = 0.0
  if len(scores)>0:
    maxS = max(scores)
    avgS = sum(scores)/float(len(scores))
  return maxS, avgS

def QueryScope(model,inv_index):
  N = 10
  if len(model)<10:
    N = len(model)
  docs = []
  for wordID,prob in sorted(model.iteritems(),\
    key=operator.itemgetter(1),reverse=True)[:N]:
    docs = list(set(docs)|set(inv_index[wordID].values()))
  return -1*math.log(float(len(docs)+1.0)/5047.0)

def idfDev(model,inv_index):
  N = 10
  if len(model)<10:
    N = len(model)
  idfs = []
  for wordID,prob in sorted(model.iteritems(),\
    key=operator.itemgetter(1),reverse=True)[:N]:
    N = float(len(inv_index[wordID]))
    idf = math.log(5047.0+0.5)/(N+1)/math.log(5047.0+1)
    idfs.append(idf)
  return stdev(idfs)

def stdev(data):
  norm2 = 0.0
  mean = sum(data)/float(len(data))
  for d in data:
    norm2 += math.pow(d,2)/float(len(data))
  if norm2-math.pow(mean,2) < 0:
    return 0;
  return math.sqrt(norm2-math.pow(mean,2))

def Variability(ret,segments):
  means = []
  vars = []
  D2 = 0.0
  D1 = 0.0
  for i in range(segments[-1]+1):
    D1 += ret[i][1]
    D2 += math.pow(ret[i][1],2)
    if i in segments:
      mean = D1/float(i)
      var = math.sqrt(D2-math.pow(mean,2))
      means.append(mean)
      vars.append(var)
  return means, vars


def Exp(lamda,x):
  return lamda*math.exp(-1*lamda*x)

def FitExpDistribution(ret,lamda):
  expdist = [Exp(lamda,x) for x in range(100)]
  mean = sum(expdist)/float(len(expdist))
  expdist = [x/mean for x in expdist]

  rank = map(operator.itemgetter(1),ret)[:100]
  meanX = sum(rank)/float(len(rank)) + rank[-1]
  if meanX == 0:
      return 0.
  rank = [(r+rank[-1])/meanX for r in rank]

  err = 0.0
  for i in range(len(rank)):
    err += math.pow(rank[i]-expdist[i],2)
  return math.sqrt(err)

def FitGaussDistribution(ret,m,v):
  gaussdist = [gaussian(m,v,x) for x in range(100)]
  mean = sum(gaussdist)/float(len(gaussdist))
  gaussdist = [x/mean for x in gaussdist]

  rank = map(operator.itemgetter(1),ret)[:100]
  meanX = sum(rank)/float(len(rank)) + rank[-1]
  if meanX == 0:
     return 0.
  rank = [(r+rank[-1])/meanX for r in rank]

  err = 0.0
  for i in range(len(rank)):
    err += math.pow(rank[i]-gaussdist[i],2)
  return math.sqrt(err)
