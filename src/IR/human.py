from util import *
from expansion import expansion

FEEDBACK_BY_DOC=0
FEEDBACK_BY_KEYTERM=1
FEEDBACK_BY_REQUEST=2
FEEDBACK_BY_TOPIC=3
SHOW_RESULT=4
ACTION_NONE=5

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
      # ans
      self.ans = ans

      # Information for actions
      self.keytermlist  = readKeytermlist(self.cpsID,query)
      self.requestlist  = readRequestlist(self.cpsID, self.ans)
      self.topiclist    = readTopicWords(self.cpsID)
      self.topicRanking = readTopicList(self.cpsID,ans_index)[:5]

    def respond(self, request ):
      ret = request[0]
      action_type = request[1]

      if action_type == 0:
        doc = next( ( item[0] for item in ret if self.ans.has_key(item[0]) ), None )
        return doc
      elif action_type == 1:
        if len(self.keytermlist):
          keyterm = self.keytermlist[0][0]
          docdir = self.dir + 'docmodel/ref/' + self.docmodeldir.split('/')[-2] + '/'
      	  cnt = sum( 1.0 for a in self.ans \
                if readDocModel(docdir + IndexToDocName(a)).has_key(a) )
          del self.keytermlist[0]
          ret =  ( True if cnt/len(self.ans) > 0.5 else False )
          return ( keyterm, ret )
        else:
          return 204
      elif action_type == 2:
        request = self.requestlist[0][0]
        del self.requestlist[0]
        return request
      elif action_type == 3:
        topic = self.topicRanking[0][0]
        del self.topicRanking[0]
        return topic
      elif action_type == 4:
        return None
      else:
        return ValueError
