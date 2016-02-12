from util import readKeytermlist, readRequestlist, readTopicWords, readTopicList

class Simulator(object):
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
    self.query = query
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

    if action == 'none':
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
        ret =  ( True if cnt/len(self.ans) > 0.5 else False )
        params['keyterm'] = [ keyterm, ret ]

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

  def view(self, params):
    self.ret = params['ret']
