
from searchengine import SearchEngine
from dialoguemanager import DialogueManager,StateMachine
from human import Simulator

# Define Cost Table
def genCostTable():
    values = [ -30., -10., -50., -20., 0., 0., 1000. ]
    costTable = dict(zip(range(6)+['lambda'],values))
    return costTable

class Environment(object):
  def __init__(self,lex,background,inv_index,doclengs,docmodeldir,dir):
    # Cost Table
    self.costTable = genCostTable()

    # Dialogue Manager, with Search Engine and StateMachine
    self.dialoguemanager = DialogueManager(
                                    lex         = lex,
                                    background  = background,
                                    inv_index   = inv_index,
                                    doclengs    = doclengs,
                                    dir         = dir,
                                    docmodeldir = docmodeldir
                                    )
    # Simulator
    self.simulator = Simulator(
                            dir         = dir,
                            docmodeldir = docmodeldir,
                            )

  def setSession(self,query,ans,ans_index):
    """
      Description:
        Sets query and answer for this session

      Return:
        state: 1 dim feature vector ( firstpass result )
    """

    self.simulator( query, ans, ans_index )

    self.dialoguemanager( query, ans )

    firstpass = self.dialoguemanager.get_retrieved_result()

    return firstpass

  def step(self, action_type):
    """
      Description:
        Has to have a query before calling this function.

      Input:
        (1) action: integer value ( >= 0 )
      Return:
        (1) State: 1 dim vector
        (2) Reward: 1 real value
    """
    assert self.dialoguemanager.posmodel != None
    assert 0 <= action_type <= 4, 'Action_type not found!'

    if action_type == 4: # Show Result
      # Terminated episode
      self.dialoguemanager.terminal = True
      # Reward is 0 and feature is None
      reward = self.costTable[ action_type ] # + self.costTable['lambda'] * (self.lastAP - self.lastAP)

      feature = None
    else:
      # Interact with Simulator
      request  = self.dialoguemanager.request(action_type)
      response = self.simulator.respond(request)

      # Expands query with simulator response
      self.dialoguemanager.expand_query(response)

      # Get state feature
      feature = self.dialoguemanager.get_retrieved_result()

      # Calculate Reward
      reward = self.dialoguemanager.calculate_reward()

    return reward,feature

  def game_over(self):
    return self.dialoguemanager.game_over()

if __name__ == "__main__":
  pass
