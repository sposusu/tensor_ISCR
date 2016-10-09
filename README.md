# Interactive Spoken Content Retrieval

### Installation
  1. Lasagne, Theano
  2. Progressbar, tqdm

### Work flow
  1. Built language models from PTV transcripts
    * Transcript directory should be a directory with T0001,T0002,...T5047 transcription files
    * Specify transcript directory in src/transcript2docmodel.py
    * Takes approximately 6 hours mainly due to 100k (keyterm)
    * cmd: python src/transcript2docmodel.py
  2. Train agent
    * run.py: Specify data, fold, feature, experiment_prefix(directory to save results), result_directory with argparser
    * Other argument can be adjusted/added/altered, see for yourself
    * cmd: python src/run.py
  3. View Results
    * Use merge_csvs.py to merge result/*.log
    * cmd: python results/merge_csvs.py $dir

### Change feature
  1. Change feature type: src/IR/statemachine.py, run_training.py
    - if/else condition in constructor, featureExtraction & argparser

### Change cost
  1. Change cost table: src/IR/actionmanager.py, possibly add another option in run_training.py, argparse

### (NOT implemented) Visualize
  1. I would record features, save to pickle of whatever during training, then use model pickle files ( stored in result/$exp_dir/$model ) to run and and record the results

### Other Notes
- Don't ask me about the code and the data storage format, it's just as it is
- I believe there are bugs in Wen's data, naming a few
  - Some keyterms/requests do not exist, can reproduce if I can access Wen's recognition transcripts
