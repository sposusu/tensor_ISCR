import operator
import pdb

from retmath import cross_entropy
from util import docNameToIndex, readFoldQueries
"""

  Move read function into search engine constructor?

"""

class SearchEngine(object):
  def __init__(self,lex,background,inv_index,doclengs,answers,dir,docmodeldir,alpha=1000,beta=0.1):
    # Initialize
    self.lex = readLex(dir+lex)
    self.background = readBackground(dir+background,self.lex)
    self.inv_index = readInvIndex(dir+inv_index)
    self.doclengs = readDocLength(dir+doclengs)
    self.answers = readAnswer(dir+answers,self.lex)

    # Document Model Directory, varies with query
    self.dir = dir
    self.docmodeldir = docmodeldir

    # Query expansion parameters
    self.alpha = alpha
    self.beta = beta

  def __call__(self,alpha=1000,beta=0.1):
    """
     Sets alpha, beta paramters for query expansion
    """
    self.alpha = alpha
    self.beta = beta

  def retrieve(self, query, negquery = None):
    """
      Retrieves result using query and negquery if negquery exists
    """
    result = {}
    for i in range(1,5048,1):
      result[i] = -9999
    # Query
    for wordID, weight in query.iteritems():
      existDoc = {}
      for docID, val in self.inv_index[wordID].iteritems():
        existDoc[docID] = 1

        # smooth doc model by background
        alpha_d = self.doclengs[docID]/(self.doclengs[docID]+self.alpha)
        qryprob = weight
        docprob = (1-alpha_d)*self.background[wordID]+alpha_d*val

        # Adds to result
        if result[docID] != -9999:
          result[docID] += cross_entropy(qryprob,docprob)
        else:
          result[docID] = cross_entropy(qryprob,docprob)

      # Run background model
      for docID, val in result.iteritems():
        if not existDoc.has_key(docID) and self.background.has_key(wordID):
          alpha_d = self.doclengs[docID] / ( self.doclengs[docID] + self.alpha )
          qryprob = weight
          docprob = (1-alpha_d) * self.background[wordID]

          if result[docID] != -9999:
            result[docID] += cross_entropy(qryprob,docprob)
          else:
            result[docID] = cross_entropy(qryprob,docprob)

    # Run through negative query
    if negquery:
      for wordID, weight in negquery.iteritems():
        existDoc = {}
        for docID, val in self.inv_index[wordID].iteritems():
          existDoc[docID] = 1
          # smooth doc model by background
          alpha_d = self.doclengs[docID]/(self.doclengs[docID]+self.alpha)
          qryprob = weight
          docprob = (1-alpha_d)*self.background[wordID]+alpha_d*val

          if result[docID] != -9999:
            result[docID] -= self.beta * cross_entropy(qryprob,docprob)
          else:
            result[docID] = -1 * self.beta * cross_entropy(qryprob,docprob)

        # Run through background model
        for docID, val in result.iteritems():
          if not existDoc.has_key(docID) and self.background.has_key(wordID):
            alpha_d = self.doclengs[docID]/(self.doclengs[docID]+self.alpha)
            qryprob = weight
            docprob = (1-alpha_d) * self.background[wordID]

          if result[docID] != -9999:
            result[docID] -= self.beta * cross_entropy(qryprob,docprob)
          else:
            result[docID] = -1 * self.beta * cross_entropy(qryprob,docprob)

    sorted_ret = sorted(result.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sorted_ret

"""
  Retrieval Engine Read Functions ( Migrated from util.py )
"""

def readLex(fname):
  lex = {}
  with open(fname) as f:
    for idx, row in enumerate(f.readlines()):
        lex[row.strip('\n')] = idx + 1
  return lex

def readList(fname):
  return [ line.strip('\n') for line in open(fname,'r').readlines() ]

def readInvIndex(fname):
  inv_index = {}
  with open(fname) as f:
    for line in f.readlines():
      [ p1, p2 ] = line.strip('\n').split('\t')
      inv_index[ int(p1) ] = dict( [ (int(valpair.split(':')[0]),float(valpair.split(':')[1])) \
                                  for valpair in p2.split() ] )
  return inv_index

def readBackground(fname,lex):
  background = {}
  with open(fname) as f:
    for line in f.readlines():
      [ p1,p2 ] = line.strip('\n').split()
      background[ lex[p1] ] = float(p2)
  return background

def readAnswer(fname,lex):
  answer = [ dict() for x in range(163) ]
  with open(fname) as f:
    for line in f.readlines():
        [t1,t2,t3,t4] = line.strip('\n').split()
        answer[ int(t1) - 1 ][ docNameToIndex(t3) ] = 1
  return answer

def readDocLength(fname):
  doclengs = {}
  with open(fname) as f:
    for line in f.readlines():
      [ p1,p2 ] = line.strip('\n').split()
      doclengs[ docNameToIndex(p1) ] = float(p2)
  return doclengs

if __name__ == "__main__":
  pass
