#from retrievalmodule import DialogueManager
#from human import SimulatedUser
import numpy as np

class Environment(object):
  def __init__(self, dialoguemanager,simulateduser):
    # Retrieval Module, with Search Engine and State Machine
    self.dialoguemanager = dialoguemanager
    # Simulated User
    self.simulateduser = simulateduser

  def setSession(self,query,ans,ans_index,test_flag = False):
    """
      Description:
        Sets query and answer for this session

      Return:
        state: 1 dim feature vector ( firstpass result )
    """
    # Sets up query and answer
    self.simulateduser( query, ans, ans_index )
    self.dialoguemanager( query, ans, test_flag ) # ans is for MAP

    # Begin first pass
    action_type = -1 # Action None

    request  = self.dialoguemanager.request( action_type )
    feedback = self.simulateduser.feedback(request)
    self.dialoguemanager.expand_query(feedback)

    firstpass = self.dialoguemanager.gen_state_feature()
    return firstpass # feature

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
      self.simulateduser.view(ret)

      # feature is None
      feature = None
    else:
      # Interact with Simulator
      request  = self.dialoguemanager.request(action_type) # wrap retrieved results & action as a request
      feedback = self.simulateduser.feedback(request)

      # Expands query with simulator response
      self.dialoguemanager.expand_query(feedback)

      # Get state feature
      feature = self.dialoguemanager.gen_state_feature()

    # Calculate Reward  (Must be retrieval reward + user reward?)
    reward = self.dialoguemanager.calculate_reward() #+ np.random.normal(0,self.reward_std)

    return reward, feature

  def game_over(self):
    return self.dialoguemanager.game_over()

if __name__ == "__main__":
  pass
