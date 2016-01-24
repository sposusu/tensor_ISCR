__all__ = ['ActionManager']

# Define Cost Table
def genCostTable():
  values = [ -30., -10., -50., -20., 0., 0., 1000. ]
  costTable = dict(zip(range(6)+['lambda'],values)
  return costTable

# Define Actions
class Action(object):
  FEEDBACK_BY_DOC      = 0
  FEEDBACK_BY_KEYTERM  = 1
  FEEDBACK_BY_REQUEST  = 2
  FEEDBACK_BY_TOPIC    = 3
  SHOW_RESULTS         = 4
  ACTION_NONE          = 5

class Request(Action):
  def __init__(self):
    pass

class Reponse(Action):
  def __init__(self):
    pass

# Define Action Manager
class ActionManager(object):
  def __init__(self):
    self.costTable = genCostTable()
    

  def __call__(self):
    pass

  def request(self):
    pass

  def respond(self):
    pass
