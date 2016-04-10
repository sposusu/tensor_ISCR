import cPickle as pickle
import pdb
import requests
import json

from api import get_simulator

def load_data(fold=-1):
  newdir = '../Data/query/'
  data = pickle.load(open(newdir+'data.pkl','r'))
  return data

host = 'http://localhost:1111/'

simulator = get_simulator()

queries = load_data()

for idx,(q, ans, ans_index) in enumerate(queries):
  print
  print 'Running query {0}'.format(idx)
  # Sets single query
  simulator(q,ans,ans_index)

  query_string = json.dumps(q)
  ans_string = json.dumps(ans)

  # First pass and set up
  firstpass_data = {'action':'firstpass','query': query_string,'ans':ans_string }

  # Posts to host
  response = requests.post(url=host+'interact',data=firstpass_data)
  res_param = response.json()

  curtHorizon = 0
  print 'Turn {0}: {1}'.format(curtHorizon,res_param['action'])

  while res_param['action'] != 'show':
    feedback_data  = simulator.feedback(res_param)
    response = requests.post(url=host+'interact',data=feedback_data)
    curtHorizon += 1
    res_param = response.json()
    print 'Turn {0}: {1}'.format(curtHorizon,res_param['action'])
    break
