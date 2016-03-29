import cPickle as pickle
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
  # Sets single query
  simulator(q,ans,ans_index)

  query_string = json.dumps(q)
  ans_string = json.dumps(ans)

  # First pass and set up
  data = {'method':'firstpass','query': query_string,'ans':ans_string,'ans_index':ans_index }

  # Posts to host
  r = requests.post(host + 'interact',data)
  assert r.json()['query'] == q

  #print idx, r.text
  break
