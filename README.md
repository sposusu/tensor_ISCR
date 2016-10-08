# Interactive Spoken Content Retrieval

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

### Feature
  1. Change feature type: src/IR/statemachine.py, run.py
    - if/else condition in constructor, featureExtraction & argparser

### Other Notes
- Don't ask me about the code and the data storage format, it's just as it is
- I believe there are bugs in old recognition, naming a few
  - Some keyterms do not exist in Wen's data, can reproduce if I can access Wen's recognition transcripts
