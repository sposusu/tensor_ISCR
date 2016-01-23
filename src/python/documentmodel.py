from util import docNameToIndex

class DocumentModel(object):
  def __init__(self,lex,background,inv_index,doclengs,answers,docmodeldir,dir):
    # Initialize
    self.lex = readLex(dir+lex)
    self.back = readBackground(dir+background,self.lex)
    self.inv_index = readInvIndex(dir+inv_index)
    self.doclengs = readDocLength(dir+doclengs)
    self.answers = readAnswer(dir+answers,self.lex)

    # Document Model Directory, varies with query
    self.dir = dir
    self.docmodeldir = docmodeldir

"""
  DocumentModel Read Functions ( Migrated from util.py )
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
