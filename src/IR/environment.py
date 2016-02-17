from dialoguemanager import DialogueManager
from human import Simulator

class Environment(object):
  def __init__(self,lex,background,inv_index,doclengs,docmodeldir,dir):
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
                            docmodeldir = docmodeldir
                            )

  def setSession(self,query,ans,ans_index,test_flag = False):
    """
      Description:
        Sets query and answer for this session

      Return:
        state: 1 dim feature vector ( firstpass result )
    """
    # Sets up query and answer
    self.simulator( query, ans, ans_index )
    print 'index :', ans_index,'query : ', query
    self.dialoguemanager( query, ans, test_flag ) # ans is for MAP

    # Begin first pass
    action_type = -1 # Action None

    request  = self.dialoguemanager.request( action_type )
    feedback = self.simulator.feedback(request)
    self.dialoguemanager.expand_query(feedback)

    firstpass = self.dialoguemanager.gen_state_feature()

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
    assert self.dialoguemanager.actionmanager.posmodel != None
    assert 0 <= action_type <= 4, 'Action_type not found!'

    if action_type == 4: # Show Result
      # Terminated episode
      ret = self.dialoguemanager.show()
      self.simulator.view(ret)

      # feature is None
      feature = None
    else:
      # Interact with Simulator
      request  = self.dialoguemanager.request(action_type)
      feedback = self.simulator.feedback(request)

      # Expands query with simulator response
      self.dialoguemanager.expand_query(feedback)

      # Get state feature
      feature = self.dialoguemanager.gen_state_feature()

    # Calculate Reward
    reward = self.dialoguemanager.calculate_reward()

    return reward, feature

  def game_over(self):
    return self.dialoguemanager.game_over()

if __name__ == "__main__":
  pass
