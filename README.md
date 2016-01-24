# InteractiveRetrieval

Our codes consists of two parts:
    Deep Q learning & Interactive Retrieval System
    
Deep Q Learning Code from https://github.com/spragunr/deep_q_rl

Interactive System(Environment) :
    1. Dialogue Manager:
        The dialogue manager that handles the interaction process 
            such as query expansion( background, negative ), records turns, calculates reward
            
        a. State Machine
            Mainly used for feature extraction
            Feature: the 1-dimensional observation vector our DQN agent needs
            
        Todo: 
            b. Action Manager 
                Performs actions that produces posmodel and negmodel for search engine retrieval
            
    2. Search Engine
        The retrieval engine of our system. 
        Inputs query and optional neg_query,
        Returns list of documents
            
    3. Simulator:
        Simulates Human behavior, ( response to human actions )
            
    Todo:
        Dataset Manager, since dataset is used universally
