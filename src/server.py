import argparse
import logging
import json
import sys

from bottle import view, get, post, run
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
@get('/')
@view('index')
def index():
  return {}

# Interact with Simulators
@post('/interact')
def interact():

  query  = request.params.get('query','')
  method = request.params.get('method','')
  return {'action':'none','query':query}
  """  Not completed
  if method == 'firstpass':
    # Sets session for current dialogue manager
    ans = request.params.get('ans','')
    ans_index = request.params.get('ans_index','')
    dialoguemanager(query,ans)
    dialoguemanager.expand_query({'action':'none','query':query})
    return {'action':'none','query':query}
  elif method == 'doc':
    doc = request.params.get('doc','')
  elif method == 'topic':
    pass
  elif method == 'keyterm':
    pass
  elif method == 'request':
    pass
  else:
    return HTTPResponse(status_code=404,body='Method not found!')

  state = dialoguemanager.gen_state_feature()
  action_type = agent.start_episode(firstpass)

  feedback_method = ActionTable[action_type]


  return 'haha'
  """

# Monitor
@get('/admin/<command>')
def admin(command):
  return 'Admin'

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-p','--port',type=int,default=1111)
  args = parser.parse_args()

  run(host='localhost', port=1111, debug=True,reloader=True)
  return 0

if __name__ == "__main__":
  sys.exit(main())
