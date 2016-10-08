# Interactive Spoken Content Retrieval

### 1. Built language models from PTV transcripts
  * Transcript directory should be a directory with T0001,T0002,...T5047 transcription files
  * Specify transcript directory in src/transcript2docmodel.py
  * python src/transcript2docmodel.py ( Takes approximately 6 hours, due to creating keyterms, you will see)


### 2. Train Retrieval agent
  * run.py: Specify data, fold, feature, experiment_prefix(directory to save results) with argparser\
  * python src/run.py
  * Other argument can be adjusted/added/altered, see for yourself

### 3. View Results
  * Use merge_csvs.py in results/

##### Notes
- Don't ask me about the code and the data storage format, it's just as it is
- I believe there are bugs in old recognition, naming a few
  - Some keyterms do not exist in Wen's data, can reproduce if I can access Wen's recognition transcripts
