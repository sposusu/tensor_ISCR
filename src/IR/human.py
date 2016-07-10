from util import *
import numpy as np
class SimulatedUser(object):
  """
    Simulates human response to retrieval machine
  """
  def __init__(self,dir, docmodeldir, keyterm_thres, topic_prob, survey):
    self.dir = dir
    self.docmodeldir = docmodeldir
    self.cpsID = '.'.join(docmodeldir.split('/')[1:-1])
    self.keyterm_thres = keyterm_thres
    self.topic_prob = topic_prob
    print "keyterm_thres = ",self.keyterm_thres

    self.ans = None

    # surveyed distribution
    self.survey = survey
    self.doc_prob = 76.78
    self.keyterm_prob = 95.28
    self.request_type = 22
    self.topic_prob = 73.14
    

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
      list = [ item[0] for item in ret if self.ans.has_key(item[0]) ]
      doc = next( ( item[0] for item in ret if self.ans.has_key(item[0]) ), None )

      if self.survey:
        flag = np.random.uniform()
        if flag < self.doc_prob:
           doc = list[0]
        else:
           doc = list[1]
           
      params['doc'] = doc

    elif action == 'keyterm':
      if len(self.keytermlist):
        keyterm = self.keytermlist[0][0]
        docdir = self.dir + 'docmodel/ref/' + self.docmodeldir.split('/')[-2] + '/'
        cnt = sum( 1.0 for a in self.ans \
          if readDocModel(docdir + IndexToDocName(a)).has_key(a) )
        del self.keytermlist[0]
        isrel =  ( True if cnt/len(self.ans) > self.keyterm_thres else False )

        if self.survey:
          flag = np.random.uniform()
          if flag > self.keyterm_prob:
            isrel = not isrel

        params['keyterm'] = keyterm
        params['isrel'] = isrel
        
    elif action == 'request':
      if not self.survey:
        request = self.requestlist[0][0]
        del self.requestlist[0]
      else:
        flag = np.random.randint(self.request_type)
        request = self.requestlist[flag][0]
        del self.requestlist[flag]
      params['request'] = request

    elif action == 'topic':
      if self.topic_prob:
        # normalize
        score = [ score*(score > 0) for topic,score in self.topicRanking ]
        if sum(score)==0:
          params['topic'] = None
        else:
          prob = [s / sum(score) for s in score ]
          choice = np.random.choice(len(self.topicRanking), 1, p=prob)[0]
          topicIdx = self.topicRanking[choice][0]
          del self.topicRanking[choice]
          params['topic'] = topicIdx
      else:

        if not survey:
          topicIdx = self.topicRanking[0][0]
          del self.topicRanking[0]
        else:
          flag = np.random.uniform()
          if flag < self.topic_prob:
            topicIdx = self.topicRanking[0][0]
            del self.topicRanking[0]
          else:
            topicIdx = self.topicRanking[1][0]
            del self.topicRanking[1]
        params['topic'] = topicIdx


    else:
      assert 0

    return params
  
  def feedback_demo(self,request,flag):
    no_input = flag # for generate survey

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

      if no_input:
        print("Choose one document: ")
      else:
        input("Choose one document: ")

      doc = next( ( item[0] for item in ret if self.ans.has_key(item[0]) ), None )
      print "Simulated Response: ", doc
      #iiter = [ item[0] for item in ret if self.ans.has_key(item[0]) ]
      #print iter[0]
      #print iter[1]
      params['doc'] = doc

    elif action == 'keyterm':
      if len(self.keytermlist):
        keyterm = self.keytermlist[0][0]
        print keyterm
        print "keyterm: ",big5map[keyterm-1].decode('big5')
        docdir = self.dir + 'docmodel/ref/' + self.docmodeldir.split('/')[-2] + '/'
        cnt = sum( 1.0 for a in self.ans \
          if readDocModel(docdir + IndexToDocName(a)).has_key(a) )
        del self.keytermlist[0]

        isrel =  ( True if cnt/len(self.ans) > 0.5 else False )
        params['keyterm'] = keyterm
        params['isrel'] = isrel
        if no_input:        
          print("Is this keyterm relevant( 0 = no / 1 = yes )? ")
        else:
          input("Is this keyterm relevant( 0 = no / 1 = yes )? ")
        print "Simulated Response: ", isrel
        
    elif action == 'request':
      for r in self.requestlist[:20]:
        print big5map[r[0]-1].decode('big5')
      request = self.requestlist[0][0]
      del self.requestlist[0]

      if no_input:
        print("Give new request: ")
      else:
        input("Give new request: ")
      params['request'] = request
      print "Simulated Response: ", big5map[request-1].decode('big5')


    elif action == 'topic':
      topicIdx = self.topicRanking[0][0]
      path = "../../ISDR-CMDP/lda/onebest.CMVN/"
      #print self.topicRanking
      topicID = [ topic for topic,score in self.topicRanking ]
      #score = [ score for topic,score in self.topicRanking ]
      #prob = [s / sum(score) for s in score ]
      #print prob
      for id in topicID:
        wordlist = open(path+str(id)).readlines()
        wordlist_sci = [s.replace("-0","-") for s in wordlist]
        sorted_list = sorted( wordlist_sci,key = lambda x:float(x.split('\t')[1]), reverse=True )
        words = [ int(sorted_list[i].split('\t')[0]) for i in xrange(10) ]
        print "Topic "+str(id)+ ": " + ','.join([big5map[word-1].decode('big5').rstrip('\n') for word in words])

      if no_input:
        print("Choose one topic:")
      else:
        input("Choose one topic:")
      del self.topicRanking[0]

      params['topic'] = topicIdx
      print "Simulated Response: ", topicIdx

    else:
      assert 0

    return params
  


  def view(self, params):
    self.ret = params['ret']
