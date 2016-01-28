import pdb

# Define Cost Table
def genCostTable():
  values = [ -30., -10., -50., -20., 0., 0., 1000. ]
  costTable = dict(zip(range(6)+['lambda'],values))
  return costTable

"""
FEEDBACK_BY_DOC      = 0
FEEDBACK_BY_KEYTERM  = 1
FEEDBACK_BY_REQUEST  = 2
FEEDBACK_BY_TOPIC    = 3
SHOW_RESULTS         = 4
ACTION_NONE          = 5

class Simulator(object):
  def __init__(self):
    pass

  def __call__(self):
    pass

# Define Actions
class ActionManager(object):
  def __init__(self,answers,dir,docmodeldir):
    self.dir         = dir
    self.docmodeldir = docmodeldir
    self.answers     = answers

    self.cpsID = '.'.join(docmodeldir.split('/')[1:-1])
    self.ans   = None

  def __call__(self,query,ans_index):
    # ans
    self.ans = self.answers[ans_index]

    # Information for actions
    self.keytermlist  = readKeytermlist(self.cpsID,query)
    self.requestlist  = readRequestlist(self.cpsID, self.ans)
    self.topiclist    = readTopicWords(self.cpsID)
    self.topicRanking = readTopicList(self.cpsID,ans_index)[:5]

  def request(self):
    pass

  def expand_query(self,response):
    pass

  def expand_query(self,response):
    '''
      Receives response from simulator
      PS: 204 is HTTP status code for empty content
    '''
    if self.cur_action == 0:
      doc = response
      if doc:
        self.posdocs.append(self.docmodeldir+IndexToDocName(doc))
      	self.poslengs.append(self.doclengs[doc])
    elif self.cur_action == 1:
      if response is not 204:
        keyterm = response[0]
        boolean = response[1]
        if boolean:
          self.posprior[ keyterm ] = 1.
        else:
          self.negprior[ keyterm ] = 1.
    elif self.cur_action == 2:
      request = response
      self.posprior[request] = 1.0
    elif self.cur_action == 3:
      topicIdx = response
      self.posdocs.append(pruneAndNormalize(self.topiclist[topicIdx],self.topicnumword))
      self.poslengs.append(self.topicleng)
    elif self.cur_action == 4:
      # This condition shouldn't happen, since we blocked this in environment.py
      assert response == None
      self.terminal = True

    posmodel = expansion(renormalize(self.posprior),self.posdocs,self.poslengs,self.background)
    negmodel = expansion(renormalize(self.negprior),self.negdocs,self.doclengs,self.background)

    return posmodel, negmodel

class Feedback_By_Keyterm(Action,Simulator):
  def __init__(self):
    pass

  def __name__(self):
    return "Feedback_By_Keyterm"

  @static_method
  def request(self):
    print self.__name__

  @static_method
  def respond(self):
    print self.__name__

class Feedback_By_Request(Action,Simulator):
  def __init__(self):
    pass
  def __name__(self):
    return "Feedback_By_Request"

class Feedback_By_Topic(Action,Simulator):
  def __init__(self):
    pass
  def __name__(self):
    return "Feedback_By_Topic"

  @static_method
  def request(self):
    print self.__name__

  @static_method
  def respond(self):
    print self.__name__

class Show_Results(Action,Simulator):
  def __init__(self):
    pass
  def __name__(self):
    return "Show_Results"

  @static_method
  def request(self):
    print self.__name__

  @static_method
  def respond(self):
    print self.__name__

class Do_Nothing(Action,Simulator):
  def __init__(self):
    pass
  def __name__(self):
    return "Do_Nothing"

  @static_method
  def request(self):
    print self.__name__

  @static_method
  def respond(self):
    print self.__name__

if __name__ == "__main__":
  pass
"""
