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
      Mainly used for feature extraction
      Feature: the 1-dimensional observation vector our RL agent needs

    * (Todo) Action Manager: <br/>
  Performs actions that produces posmodel and negmodel for search engine retrieval
  * Search Engine: <br/>
  The retrieval engine of our system.
  Inputs query,returns list of documents
  * Simulator: <br/>
  Simulates Human behavior, ( response to human actions )
  * (Todo) Dataset Manager
