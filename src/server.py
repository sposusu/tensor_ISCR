"""

  This is our Interactive Retrieval System's server

"""
import argparse
import logging
import json
import sys

from flask import Flask
from flask import request, abort
from flask import render_template, jsonify
import jieba

import api
from IR.actionmanager import genActionTable

app = Flask(__name__,template_folder='views')

ActionTable = genActionTable()
dialoguemanager = api.get_dialoguemanger()
agent = api.get_agent()
agent.start_testing()

# Index Page
@app.route('/')
def index():
  return render_template('index.html')

# Query with chinese unicode characters, no interactions
@app.route('/query',methods=['POST'])
def query():
  action = request.form.get('action','')
  assert action == 'firstpass'
  query_uni = request.form.get('query','')
  
  words_uni = [ w for w in jieba.cut(query_uni,cut_all=False) ]

  words_big5 = [ utf8tobig5hex(w) for w in words_uni ]

  query = big5list_to_dict(words_big5)

  return jsonify(**query)

def bracket_word(chars_big5):
  assert len(chars_big5) % 4 == 0
  bracketed_chars = ''
  for i in range(0,len(chars_big5),4):
    bracketed_chars += '[' + chars_big5[i:i+4] + ']'
  return bracketed_chars


def big5list_to_dict(words_big5):
  d = dict()
  L = len(words_big5)
  for w in words_big5:
    d[ bracket_word(w) ] = 1./L
  return d

def utf8tobig5hex(uni_string):
  big5_string = uni_string.encode('big5')
  big5hex = ''
  for c in big5_string:
    big5hex += format(ord(c),'02X')
  return big5hex

# Interact with Simulators
@app.route('/interact',methods=['POST'])
def interact():
  action = request.form.get('action','')

  if action == 'firstpass':
    # Sets session for current dialogue manager
    query  = request.form.get('query','')
    query = toint(json.loads(query))

    ans = request.form.get('ans','')
    ans = toint(json.loads(ans))

    ans_index = request.form.get('ans_index','')

    assert check_int_dict(query) and check_int_dict(ans)

    dialoguemanager(query,ans, test_flag=True)
    dialoguemanager.expand_query({'action':'firstpass','query':query})

  elif action in ['doc','topic','keyterm','request']:
    feedback_data = dict(request.form.items())
    dialoguemanager.expand_query(feedback_data)

  else:
    abort(404)
    #return HTTPResponse(status_code=404, body='Action {0} not found!'.format(action))

  # Generate State for Agent
  state = dialoguemanager.gen_state_feature()
  # Agent Part
  if action == 'firstpass':
    action_type = agent.start_episode(state)
  else:
    reward = dialoguemanager.calculate_reward()
    action_type = agent.step(reward,state)

  # Set response form
  form = dialoguemanager.request(action_type)

  return jsonify(**form)

def toint(d):
  return dict( [ (int(k),float(v)) for k,v in d.iteritems() ] )
def check_int_dict(d):
  return all(isinstance(k,int) for k in d.keys()) and all(isinstance(v,float) for v in d.values())


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-p','--port',type=int,default=1111)
  args = parser.parse_args()
  app.run(host='localhost', port=1111, debug=True)
  return 0

if __name__ == "__main__":
  sys.exit(main())
