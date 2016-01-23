from actionmanager import ActionManager
from dialoguemanager import DialogueManager,StateMachine
from human import Simulator
from searchengine import SearchEngine

class Environment(object):
  def __init__(self,lex,background,inv_index,doclengs,answers,docmodeldir,dir):
    # Search Engine
    self.searchengine = SearchEngine(
                                lex = lex,
                                background  = background,
                                inv_index   = inv_index,
                                doclengs    = doclengs,
                                answers     = answers,
                                )

    # Dialogue Manager, with State Machine for Feature Extraction
    self.dialoguemanager = DialogueManager(
                                    background  = self.searchengine.background,
                                    inv_index   = self.searchengine.inv_index,
                                    doclengs    = self.searchengine.doclengs,
                                    answers     = self.searchengine.answers,
                                    dir         = dir,
                                    docmodeldir = docmodeldir
                                    )
    # Define Action Manager
    self.actionmanager = ActionManager(

                                  )

    # Simulator
    self.simulator = Simulator(
                            answers     = self.searchengine.answers
                            dir         = dir,
                            docmodeldir = docmodeldir,
                            )

  def setSession(self,query,ans_index):
    """
      Description:
        Sets query and answer for this session

      Return:
        state: 1 dim feature vector ( firstpass result )
    """

    self.simulator(ans_index)

    self.dialoguemanager( query = query, ans_index = ans_index )
    self.dialoguemanager.get_retrieved_result( ret = self.searchengine.retrieve(query) )

    return self.dialoguemanager.featureExtraction()

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
    assert self.query != None
    assert 0 <= action_type <= 4, 'Action_type not found!'

    if action_type == 4: # Show Result
      # Terminated episode
      self.terminal = True
      # Reward is 0 and feature is None
      reward = self.costTable[ action_type ] # + self.costTable['lambda'] * (self.lastAP - self.lastAP)
      feature = None
    else:
      # Interact with Simulator
      prev_ret, action_type = self.dialoguemanager.request(action_type)
      response = self.simulator.respond(prev_ret,action_type)

      # Expands query with simulator response
      posmodel, negmodel = self.dialoguemanager.expand_query(response)

      # Search Engine retrieves results
      ret = self.searchengine.retrieve(posmodel,negmodel)

      # Set retrieved result to dialogue manager
      self.dialoguemanager.get_retrieved_result(ret)

      # Get state feature
      feature = self.dialoguemanager.featureExtraction()

      # Calculate Reward
      reward = self.dialoguemanager.calculate_reward()
      '''
      reward = self.costTable[action_type] +  \
              self.costTable['lambda'] * self.dialoguemanager.APincrease()
      '''
    return reward,feature

  def game_over(self):
    if self.dialoguemanager.curtHorizon >= 5 or self.terminal:
        self.query = None
        self.ans_index = None
        return True
    return False

if __name__ == "__main__":
  pass
