from util import *

class SimulatedUser(object):
  """
    Simulates human response to retrieval machine
  """
  def __init__(self,dir,docmodeldir):
    self.dir = dir
    self.docmodeldir = docmodeldir
    self.cpsID = '.'.join(docmodeldir.split('/')[1:-1])

    self.ans = None

  def __call__(self,query,ans,ans_index):
    # query and desired answer
    self.query = query.copy()
    self.ans   = ans
    self.ret   = None

    # Information for actions
    self.keytermlist  = readKeytermlist(self.cpsID,query)
    self.requestlist  = readRequestlist(self.cpsID, self.ans)
    self.topiclist    = readTopicWords(self.cpsID)
    self.topicRanking = readTopicList(self.cpsID,ans_index)[:5]

    # Debug
    #print 'Simualtor has query {0}, ans {1}'.format(self.query,self.ans)

  def feedback(self, request ):
    ret    = request['ret']
    action = request['action']

    params = {}
    params['action'] = action

    if action == 'firstpass':
      params['query'] = self.query

    elif action == 'doc':
      doc = next( ( item[0] for item in ret if self.ans.has_key(item[0]) ), None )
      params['doc'] = doc

    elif action == 'keyterm':
      if len(self.keytermlist):
        keyterm = self.keytermlist[0][0]
        docdir = self.dir + 'docmodel/ref/' + self.docmodeldir.split('/')[-2] + '/'
        cnt = sum( 1.0 for a in self.ans \
          if readDocModel(docdir + IndexToDocName(a)).has_key(a) )
        del self.keytermlist[0]
        isrel =  ( True if cnt/len(self.ans) > 0.5 else False )
        params['keyterm'] = keyterm
        params['isrel'] = isrel
        
    elif action == 'request':
      request = self.requestlist[0][0]
      del self.requestlist[0]
      params['request'] = request

    elif action == 'topic':
      topicIdx = self.topicRanking[0][0]
      del self.topicRanking[0]
      params['topic'] = topicIdx

    else:
      assert 0

    return params
  
  def feedback_demo(self,request):
    f = open("../../ISDR-CMDP/PTV.big5.lex","r")
    big5map = f.readlines()

    ret    = request['ret']
    action = request['action']

    params = {}
    params['action'] = action

    if action == 'firstpass':
      params['query'] = self.query

    elif action == 'doc':
      docID = [ doc for doc,score in ret[:5] ]
      path = "../../PTV/ptv_recognition_doc/doc_onebest"
      docs = [ open(path+"/T"+str(id).zfill(4)) for id in docID ]
      for i in xrange(len(docs)):
        print str(docID[i])+". "+docs[i].readline()
      params['doc'] = input("Choose one document: ")

      doc = next( ( item[0] for item in ret if self.ans.has_key(item[0]) ), None )
      print "Simulated Response: ", doc
      params['doc'] = doc

    elif action == 'keyterm':
      if len(self.keytermlist):
        keyterm = self.keytermlist[0][0]
        print big5map[keyterm-1].decode('big5')
        docdir = self.dir + 'docmodel/ref/' + self.docmodeldir.split('/')[-2] + '/'
        cnt = sum( 1.0 for a in self.ans \
          if readDocModel(docdir + IndexToDocName(a)).has_key(a) )
        del self.keytermlist[0]
        isrel =  ( True if cnt/len(self.ans) > 0.5 else False )
        params['keyterm'] = keyterm
        params['isrel'] = isrel
        ans = input("Is this keyterm relevant( 0 = no / 1 = yes )? ")
        print "Simulated Response: ", isrel
        
    elif action == 'request':
      request = self.requestlist[0][0]
      del self.requestlist[0]

      input("Give new request: ")
      params['request'] = request
      print "Simulated Response: ", big5map[request-1].decode('big5')


    elif action == 'topic':
      topicIdx = self.topicRanking[0][0]
      path = "../../ISDR-CMDP/lda/onebest.CMVN/"
      topicID = [ topic for topic,score in self.topicRanking ]
      for id in topicID:
        wordlist = open(path+str(id)).readlines()
        wordlist_sci = [s.replace("-0","-") for s in wordlist]
        sorted_list = sorted( wordlist_sci,key = lambda x:float(x.split('\t')[1]), reverse=True )
        words = [ int(sorted_list[i].split('\t')[0]) for i in xrange(10) ]
        print words
        print "Topic "+str(id)+ ": " + ','.join([big5map[word-1].decode('big5').rstrip('\n') for word in words])
      input("Choose one topic:")
      del self.topicRanking[0]

      params['topic'] = topicIdx
      print "Simulated Response: ", topicIdx

    else:
      assert 0

    return params
  


  def view(self, params):
    self.ret = params['ret']
