# InteractiveRetrieval

Our codes consists of two parts:
    * Deep Reinforcment Learning 
    * Interactive Retrieval System
    
Deep Reinforcement Learning Code:
    * https://github.com/spragunr/deep_q_rl

Interactive System(Environment) :
    * Dialogue Manager:
        The dialogue manager that handles the interaction process 
            - query expansion( background, negative )
            - records turns
            - calculates reward
        * State Machine:
            Mainly used for feature extraction
                - Feature: the 1-dimensional observation vector our RL agent needs
        * (Todo) Action Manager: 
            Performs actions that produces posmodel and negmodel for search engine retrieval
    * Search Engine
        The retrieval engine of our system. 
        Inputs query and optional neg_query,
        Returns list of documents
    * Simulator:
        Simulates Human behavior, ( response to human actions )
    * (Todo) Dataset Manager
