# InteractiveRetrieval

Our codes consists of two parts:
  * Deep Reinforcment Learning
  * Interactive Retrieval System

Deep Reinforcement Learning Code:
  * https://github.com/spragunr/deep_q_rl

Interactive System(Environment): <br/>
  * Dialogue Manager: <br/>
      The dialogue manager that handles the interaction process
    * Roles
      - query expansion(background model, negative model)
      - records turns
      - calculates reward

    * State Machine: <br/>
      Extracts feature and performs state estimation
      Feature: the 1-dimensional observation vector our RL agent needs

    * Action Manager: <br/>
      Performs query expansion according to different actions

  * Search Engine: <br/>
    The retrieval engine of our system.
    Inputs query,returns list of documents
  * Simulator: <br/>
    Simulates Human behavior, ( response to human actions )

DeepReinforce.py
  * agent.py(in folder DQN)
    * q_network.py
      * lasagne_updates.py
    * ale_data_set.py 
                      
  * environment.py (in folder IR)
    * dialoguemanager.py(retrieval module)
                                           -- seachengin.py 
                                           -- statemachine.py
                                           -- actionmanager.py
    * human.py (simulated user)
