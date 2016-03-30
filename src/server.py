import argparse
import logging
import json
import sys

from bottle import route, view, get, post, run
from bottle import request, response, HTTPResponse
from bottle import redirect

import api
from IR.actionmanager import genActionTable
from IR.dialoguemanager import DialogueManager
"""

  This is our Interactive Retrieval System's server

"""
ActionTable = genActionTable()


dialoguemanager = api.get_dialoguemanger()
agent = api.get_agent()

agent.start_testing()

# Index Page
@route('/')
@view('index')
def index():
  return {}

# Interact with Simulators
@post('/interact')
def interact():
  action = request.params.get('action','')

  if action == 'firstpass':
    # Sets session for current dialogue manager
    query  = request.params.get('query','')
    query = toint(json.loads(query))

    ans = request.params.get('ans','')
    ans = toint(json.loads(ans))

    ans_index = request.params.get('ans_index','')

    assert check_int_dict(query) and check_int_dict(ans)

    dialoguemanager(query,ans, test_flag=True)
    dialoguemanager.expand_query({'action':'firstpass','query':query})

  elif action in ['doc','topic','keyterm','request']:
    feedback_data = dict(request.params.items())
    dialoguemanager.expand_query(feedback_data)

  else:
    return HTTPResponse(status_code=404, body='Action {0} not found!'.format(action))

  # Generate State for Agent
  state = dialoguemanager.gen_state_feature()

  # Agent Part
  if action == 'firstpass':
    action_type = agent.start_episode(state)
  else:
    reward = dialoguemanager.calculate_reward()
    action_type = agent.step(reward,state)

  # Set response params
  params = dialoguemanager.request(action_type)

  return params

# Monitor
@get('/admin/<command>')
def admin(command):
  return 'Admin'

def toint(d):
  return dict( [ (int(k),float(v)) for k,v in d.iteritems() ] )
def check_int_dict(d):
  return all(isinstance(k,int) for k in d.keys()) and all(isinstance(v,float) for v in d.values())

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-p','--port',type=int,default=1111)
  args = parser.parse_args()

  run(host='localhost', port=1111, debug=True, reloader=True)

  return 0

if __name__ == "__main__":
  sys.exit(main())
